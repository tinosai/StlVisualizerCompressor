#!/usr/bin/env python
import sys
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import glob
from natsort import natsorted
from torch.utils.data import DataLoader
from dotenv import load_dotenv
import datetime
load_dotenv("config.env")


# 1. Set up the path so that the model can be imported
# path where the current script is located
current_path = os.path.dirname(__file__)
# path to the parent directory for the scripts to load
utilities_path = os.path.abspath(os.path.join(current_path, os.pardir, "utilities"))
print(f"Adding {utilities_path} to the system path")
sys.path.append(utilities_path)
from randLaNetModel import *

# 2. Load the setup for the model from the environment variables. Configured in the config.env file
model_settings = {}
model_settings['device'] = os.environ.get("MODEL_device")
model_settings["d_in"] = int(os.environ.get("MODEL_d_in"))
model_settings["d_out"] = int(os.environ.get("MODEL_d_out"))
model_settings["num_neighbors"] = int(os.environ.get("MODEL_num_neighbors"))
model_settings["decimation"] = int(os.environ.get("MODEL_decimation"))
device = model_settings["device"]


# 3. Instatntiate the model. The device to be used is set in the environment variable
model = RandLANetModel(**model_settings)

# 4. Create the point cloud data set
class PointCloudDataset(Dataset):
    def __init__(self, points_dir, targets_dir):
        self.points_paths = natsorted(glob.glob(os.path.join(points_dir, "*.txt")))
        self.targets_paths = natsorted(glob.glob(os.path.join(targets_dir, "*.txt")))

        assert len(self.points_paths) == len(self.targets_paths), "Mismatch between points and targets!"

    def __len__(self):
        return len(self.points_paths)

    def __getitem__(self, idx):
        # Read files
        x = np.loadtxt(self.points_paths[idx])  # shape (N, d_in)
        y = np.loadtxt(self.targets_paths[idx]) # shape (N, d_out)
        if len(y.shape) == 1:
            y = y[:, np.newaxis]

        # Pack into dict
        return {
            'points': torch.tensor(x, dtype=torch.float32),  # (N, d_in)
            'targets': torch.tensor(y, dtype=torch.float32)  # (N, d_out)
        }

# 5. Set up the data loader which pads the clouds to the same size.
def pad_and_mask(batch):
    # batch = list of dicts
    points_list = [item['points'] for item in batch]
    targets_list = [item['targets'] for item in batch]

    max_pts = max(p.shape[0] for p in points_list)

    batch_points = []
    batch_targets = []
    batch_mask = []

    for p, t in zip(points_list, targets_list):
        n_pts = p.shape[0]

        # Padding
        p_padded = torch.zeros((max_pts, p.shape[1]), dtype=p.dtype)
        p_padded[:n_pts] = p

        t_padded = torch.zeros((max_pts, t.shape[1]), dtype=t.dtype)
        t_padded[:n_pts] = t

        mask = torch.zeros(max_pts, dtype=torch.bool)
        mask[:n_pts] = True

        batch_points.append(p_padded)
        batch_targets.append(t_padded)
        batch_mask.append(mask)

    # Stack into batch
    batch_points = torch.stack(batch_points)  # (B, max_pts, d_in)
    batch_targets = torch.stack(batch_targets)  # (B, max_pts, d_out)
    batch_mask = torch.stack(batch_mask)  # (B, max_pts)

    return batch_points, batch_targets, batch_mask


# 6. Instantiate the dataset, dataloader and optimizer
ds = PointCloudDataset(os.path.join(current_path, "../data", "points_and_normals"), os.path.join(current_path, "../data", "dummy_data"))
loader = DataLoader(ds, batch_size=int(os.environ["TRAINING_BATCH_SIZE"]), collate_fn=pad_and_mask, shuffle=True, num_workers=0)
optimizer = getattr(torch.optim, os.environ["TRAINING_OPTIMIZER"])(model.parameters(), lr=float(os.environ["TRAINING_LR"]))

if not os.path.exists(os.path.join(current_path, "../models")):
    os.makedirs(os.path.join(current_path, "../models"))
model_path = os.path.join(current_path, "../models", f"randlanet_model_{os.environ['TRAINING_EPOCHS']}.pth")


if not os.path.isfile(model_path):
    print("Model file does not exist. Training a new model.")
    # 7. Train the model
    for epoch in range(int(os.environ["TRAINING_EPOCHS"])):
        for i, batch in enumerate(loader):
            points = batch[0].to(device)
            targets = batch[1].to(device)
            mask = batch[2].to(device)

            # Forward pass
            pred = model(points, mask)

            # Compute loss
            loss = torch.nn.functional.mse_loss(pred[mask], targets[mask])

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            print(f"{datetime.datetime.now()}: Epoch {epoch}, Batch {i}, Loss: {loss.item()}")
    # 8. Save the model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
else:  
    print("Model file exists. Loading the model.")
    # Load the model
    model.load_state_dict(torch.load(model_path))

# 9. Test the model
# Load the model
model = RandLANetModel(**model_settings)
model.load_state_dict(torch.load(model_path))
model.eval()


# 10. Test the model on a single point cloud
with torch.no_grad():
    test_points = np.loadtxt(os.path.join(current_path, "../data", "test", "dummy_file_2.txt"))  # shape (N, d_in)

    points_test = torch.tensor(test_points, dtype=torch.float32).to(device)
    mask_test = torch.ones(points_test.shape[0], dtype=torch.bool).to(device)
    pred_test = model(points_test.unsqueeze(0), mask_test.unsqueeze(0))
    
    print(pred_test)