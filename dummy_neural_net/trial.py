#!/usr/bin/env python
import sys
import os
import torch

# path where the current script is located
current_path = os.path.dirname(__file__)
# path to the parent directory for the scripts to load
utilities_path = os.path.abspath(os.path.join(current_path, os.pardir, "utilities"))
print(f"Adding {utilities_path} to the system path")
sys.path.append(utilities_path)
from randLaNetModel import *


# automatically detect the device
model_settings = {}
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

model_settings['device'] = device

model_settings["d_in"] = 6 # 3 for the coordinates, 3 for the normals
model_settings["d_out"] = 1 # just one variable to predict on each point
model_settings["num_neighbors"] = 16
model_settings["decimation"] = 4 # 4 times less points in the output. Using this as a starting point for iteration.


# instantiate the model
model = RandLANetModel(**model_settings)

# # sanity check of the dimensions
# n_in = 10000
# cloud = torch.randn((2,n_in,6)).to(device)
# mask = torch.ones((2,n_in), dtype=torch.bool).to(device)
# mask[:, cloud.shape[1]//2:] = False
# out = model(cloud, mask)
# print(f"For shape {cloud.shape} got output of shape {out.shape}")

# n_in = 60000
# cloud = torch.randn((2,n_in,6)).to(device)
# mask = torch.ones((2,n_in), dtype=torch.bool).to(device)
# mask[:, cloud.shape[1]//2:] = False
# out = model(cloud, mask)
# print(f"For shape {cloud.shape} got output of shape {out.shape}")

# here we are going to test the model with some real inputs (on a dummy dataset)
# imagine there is a training folder, which contains two subfolders: points/normals (the input data) and the value to be predicted at the point 


