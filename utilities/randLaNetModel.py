# this code is imported from the original code by the authors of RandLANet at https://github.com/aRI0U/RandLA-Net-pytorch/tree/master
# Since the original code uses libraries which are now deprecated, we have reimplemented knn to find the nearest neighbors.
# The original paper is "RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds" by Hu et al. 2020
# @article{RandLA-Net,
#   arxivId = {1911.11236},
#   author = {Hu, Qingyong and Yang, Bo and Xie, Linhai and Rosa, Stefano and Guo, Yulan and Wang, Zhihua and Trigoni, Niki and Markham, Andrew},
#   eprint = {1911.11236},
#   title = {{RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds}},
#   url = {http://arxiv.org/abs/1911.11236},
#   year = {2019}
# }
# In this code, we also included the use of a mask to ignore certain points in the point cloud, which is useful when the considered point clouds have a variable number of points.
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

def knn(pos_support, pos_query, k, mask_support=None, mask_query=None):
    """
    Performs k-nearest neighbors search between support and query points in batched 3D point clouds.
    Args:
        pos_support (torch.Tensor): Support points tensor of shape (B, N, 3) where B is batch size,
            N is number of support points, and 3 is the dimensionality.
        pos_query (torch.Tensor): Query points tensor of shape (B, M, 3) where M is number of query points.
        k (int): Number of nearest neighbors to find.
        mask_support (torch.Tensor, optional): Boolean mask for support points of shape (B, N).
            Defaults to None.
        mask_query (torch.Tensor, optional): Boolean mask for query points of shape (B, M).
            Defaults to None.
    Returns:
        tuple: Contains:
            - idx (torch.Tensor): Indices of k-nearest neighbors of shape (B, M, k)
            - dist (torch.Tensor): Distances to k-nearest neighbors of shape (B, M, k)
    Note:
        Uses scipy's cKDTree for efficient neighbor search, temporarily moving data to CPU.
        Returns -1 indices and inf distances for invalid/masked points.
    
    Note: this function is still in development. I expect to update it in the future to use FAISS or another more efficient library.
    This is just my current implementation. Most of the remaining code is from the original authors of RandLANet.
    """
    B, N, _ = pos_support.shape
    _, M, _ = pos_query.shape
    device = pos_support.device  
    
    idx = torch.full((B, M, k), fill_value=-1, dtype=torch.long, device=device)
    dist = torch.full((B, M, k), fill_value=float('inf'), dtype=torch.float, device=device)

    for b in range(B):
        support = pos_support[b]
        query = pos_query[b]

        if mask_support is not None:
            support = support[mask_support[b]]
        if mask_query is not None:
            query = query[mask_query[b]]
        
        # Still need CPU for scipy
        tree = cKDTree(support.cpu().numpy())
        dist_b, idx_b = tree.query(query.cpu().numpy(), k=k)

        if k == 1:
            idx_b = idx_b[:, None]
            dist_b = dist_b[:, None]

        valid_query_idx = mask_query[b].nonzero(as_tuple=False).squeeze(-1) if mask_query is not None else torch.arange(M, device=device)

        idx[b, valid_query_idx] = torch.from_numpy(idx_b).to(device)
        dist[b, valid_query_idx] = torch.from_numpy(dist_b).float().to(device)

    return idx, dist


class SharedMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        transpose=False,
        padding_mode='zeros',
        bn=False,
        activation_fn=None
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding_mode=padding_mode
        )
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):
        r"""
            Forward pass of the network

            Parameters
            ----------
            input: torch.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            torch.Tensor, shape (B, d_out, N, K)
        """
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class LocalSpatialEncoding(nn.Module):
    def __init__(self, d, num_neighbors, device):
        super(LocalSpatialEncoding, self).__init__()

        self.num_neighbors = num_neighbors
        self.mlp = SharedMLP(10, d, bn=True, activation_fn=nn.ReLU())

        self.device = device


    def forward(self, coords, features, knn_output, mask):
        idx, dist = knn_output
        B, N, K = idx.size()
    
        # 🛠 Fix -1 indices
        idx = idx.clamp(min=0)
    
        extended_idx = idx.unsqueeze(1).expand(B, 3, N, K)
        extended_coords = coords.transpose(-2, -1).unsqueeze(-1).expand(B, 3, N, K)
        neighbors = torch.gather(extended_coords, 2, extended_idx)  # (B, 3, N, K)
    
        if mask is not None:
            valid_mask = mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, N, 1)

            # Only mask distances, not coordinates
            dist = dist.masked_fill(valid_mask.squeeze(1) == 0, float('inf'))

    
        concat = torch.cat((
            extended_coords,
            neighbors,
            extended_coords - neighbors,
            dist.unsqueeze(-3)
        ), dim=-3).to(self.device)
    
        return torch.cat((
            self.mlp(concat),
            features.expand(B, -1, N, K)
        ), dim=-3)





class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()

        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.Softmax(dim=-2)
        )
        self.mlp = SharedMLP(in_channels, out_channels, bn=True, activation_fn=nn.ReLU())

    def forward(self, x, mask=None):
        r"""
            Forward pass

            Parameters
            ----------
            x: torch.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            torch.Tensor, shape (B, d_out, N, 1)
        """
        # computing attention scores
        scores = self.score_fn(x.permute(0,2,3,1)).permute(0,3,1,2)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, N, 1)
            scores = scores * mask
        
        # sum over the neighbors
        features = torch.sum(scores * x, dim=-1, keepdim=True) # shape (B, d_in, N, 1)

        return self.mlp(features)



class LocalFeatureAggregation(nn.Module):
    def __init__(self, d_in, d_out, num_neighbors, device):
        super(LocalFeatureAggregation, self).__init__()

        self.num_neighbors = num_neighbors

        self.mlp1 = SharedMLP(d_in, d_out//2, activation_fn=nn.LeakyReLU(0.2))
        self.mlp2 = SharedMLP(d_out, 2*d_out)
        self.shortcut = SharedMLP(d_in, 2*d_out, bn=True)

        self.lse1 = LocalSpatialEncoding(d_out//2, num_neighbors, device)
        self.lse2 = LocalSpatialEncoding(d_out//2, num_neighbors, device)

        self.pool1 = AttentivePooling(d_out, d_out//2)
        self.pool2 = AttentivePooling(d_out, d_out)

        self.lrelu = nn.LeakyReLU()

    def forward(self, coords, features, mask):
        r"""
            Forward pass

            Parameters
            ----------
            coords: torch.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: torch.Tensor, shape (B, d_in, N, 1)
                features of the point cloud

            Returns
            -------
            torch.Tensor, shape (B, 2*d_out, N, 1)
        """

        knn_output = knn(coords.contiguous(), coords.contiguous(), self.num_neighbors, mask, mask)
        
        # Important! Masked coords for safe gathering
        coords_masked = coords * mask.unsqueeze(-1).float()
        
        x = self.mlp1(features)
        
        x = self.lse1(coords_masked, x, knn_output, mask)
        x = self.pool1(x, mask)
        
        x = self.lse2(coords_masked, x, knn_output, mask)
        x = self.pool2(x, mask)
        
        return self.lrelu(self.mlp2(x) + self.shortcut(features))




class RandLANetModel(nn.Module):
    def __init__(self, d_in, d_out, num_neighbors=16, decimation=4, device=torch.device('cpu')):
        super(RandLANetModel, self).__init__()
        self.num_neighbors = num_neighbors
        self.decimation = decimation

        self.fc_start = nn.Linear(d_in, 8)
        self.bn_start = nn.Sequential(
            nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
            nn.LeakyReLU(0.2)
        )

        # encoding layers
        self.encoder = nn.ModuleList([
            LocalFeatureAggregation(8, 16, num_neighbors, device),
            LocalFeatureAggregation(32, 64, num_neighbors, device),
            LocalFeatureAggregation(128, 128, num_neighbors, device),
            LocalFeatureAggregation(256, 256, num_neighbors, device)
        ])

        self.mlp = SharedMLP(512, 512, activation_fn=nn.ReLU())

        # decoding layers
        decoder_kwargs = dict(
            transpose=True,
            bn=True,
            activation_fn=nn.ReLU()
        )
        self.decoder = nn.ModuleList([
            SharedMLP(1024, 256, **decoder_kwargs),
            SharedMLP(512, 128, **decoder_kwargs),
            SharedMLP(256, 32, **decoder_kwargs),
            SharedMLP(64, 8, **decoder_kwargs)
        ])

        # final semantic prediction
        self.fc_end = nn.Sequential(
            SharedMLP(8, 64, bn=True, activation_fn=nn.ReLU()),
            SharedMLP(64, 32, bn=True, activation_fn=nn.ReLU()),
            nn.Dropout(),
            SharedMLP(32, d_out)
        )
        self.device = device

        self = self.to(device)

    def forward(self, input, mask):
        r"""
            Forward pass

            Parameters
            ----------
            input: torch.Tensor, shape (B, N, d_in)
                input points

            Returns
            -------
            torch.Tensor, shape (B, d_out, N)
                segmentation scores for each point
        """
        N = input.size(1)
        d = self.decimation

        coords = input[...,:3].clone()#.cpu()
        x = self.fc_start(input).transpose(-2,-1).unsqueeze(-1)
        x = self.bn_start(x) # shape (B, d, N, 1)

        decimation_ratio = 1

        # <<<<<<<<<< ENCODER
        x_stack = []

        permutation = torch.randperm(N)
        coords = coords[:,permutation]
        mask = mask[:, permutation]
        x = x[:,:,permutation]

        for lfa in self.encoder:
            # at iteration i, x.shape = (B, N//(d**i), d_in)
            x = lfa(coords[:,:N//decimation_ratio], x, mask[:,:N//decimation_ratio])
            x_stack.append(x.clone())
            decimation_ratio *= d
            x = x[:,:,:N//decimation_ratio]


        # # >>>>>>>>>> ENCODER

        x = self.mlp(x)

        # <<<<<<<<<< DECODER
        for mlp in self.decoder:
            neighbors, _ = knn(
                            coords[:,:N//decimation_ratio].cpu().contiguous(),
                            coords[:,:d*N//decimation_ratio].cpu().contiguous(),
                            1,
                            mask_support=mask[:,:N//decimation_ratio].cpu(),
                            mask_query=mask[:,:d*N//decimation_ratio].cpu()
                        )
            neighbors = neighbors.to(self.device)

            extended_neighbors = neighbors.unsqueeze(1).expand(-1, x.size(1), -1, 1)

            x_neighbors = torch.gather(x, -2, extended_neighbors)

            x = torch.cat((x_neighbors, x_stack.pop()), dim=1)

            x = mlp(x)

            decimation_ratio //= d

        # >>>>>>>>>> DECODER
        # inverse permutation
        x = x[:,:,torch.argsort(permutation)]

        scores = self.fc_end(x)

        return scores.squeeze(-1).permute(0,2,1)


# if __name__ == '__main__':
#     if torch.cuda.is_available():
#         device = torch.device('cuda:0')
#     elif torch.mps.is_available():
#         device = torch.device('mps')
#     else:
#         device = torch.device('cpu')
    
#     d_in = 6
#     cloud = 1000*torch.randn(1, 2**16, d_in).to(device)
#     mask = torch.ones((cloud.shape[0], cloud.shape[1])).bool().to(device)
#     model = RandLANet(d_in = d_in, d_out = 1, num_neighbors=16, decimation=2, device=device)
#     # model.load_state_dict(torch.load('checkpoints/checkpoint_100.pth'))
#     model.eval()

#     t0 = time.time()
#     pred = model(cloud, mask)
#     t1 = time.time()
#     # print(pred)
#     print(t1-t0)