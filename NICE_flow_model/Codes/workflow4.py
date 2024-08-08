#Import libraires
import processing
import externalize as ext
import final_AE

import torch
import pandas as pd
import numpy as np
import os
import random
import time
import torch.nn as nn
import zuko
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import statistics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

#Clean and Process data
#Initiate the processor
mpc = processing.Processor_cond()

#Load raw data, this loads the data from the files and processes it to the correct format
Galaxies_raw = mpc.get_data(r"/export/home/ksampath/my_venv/Dataset/galaxy_numpy_data")

#Clean data with specific algorithm, set percentile1 and percentile2 to 98%
Galaxies_cleaned, Galaxies_removed, Stars_removed = mpc.constraindata(Galaxies_raw, percentile1=98, percentile2=98, N_min=500)
print("Data cleaned")

#Pring the galaxies removed and remove these galaxies from main dataset of images before training the AE
#Remove the above galaxies from the dataset before training autoencoder model
path = r"/export/home/ksampath/my_venv/26-07/Dataset/galaxy_image_data"
gal_rem_l = []
for galaxy_dict in Galaxies_removed:
    file_name = path+"/"+ galaxy_dict["galaxy"]["NIHAO_id"] + ".png"
    try: 
        os.remove(file_name)
        print("File removed")
    except:
        print("File not present")
    gal_rem_l.append(galaxy_dict["galaxy"]["NIHAO_id"])

#Train AE
"""#Call and train model
path = r"/export/home/ksampath/my_venv/26-07/Dataset/galaxy_image_data"
final_AE.AutoEncoder(path)"""

Galaxies_sorted = sorted(Galaxies_cleaned, key=lambda x: x['galaxy']['NIHAO_id'])
for i in Galaxies_sorted:
    i["stars"].reset_index(drop = True, inplace = True)

#Chose a subset of the data
#1. Define conditions to be used
#Done by supplying a function, that computes conditions from a galaxy (dict as above) and returns them with condition names (dict or DataFrame)
cond_fn = ext.cond_M_stars_2age_avZ

#2. Name the components to be used
#Here ignore a stars mass as they are all equal due to simulation restrictions
comp_use = ["x", "y", "z", "vx", "vy", "vz", "Z", "feh", "ofe", "age"]

#3. Define the subset to be used (MWs, all, etc.)
# Exclude specific numbers from the random sample
exclude = {66, 20, 33, 48, 5}
# Generate a list of 75 unique random numbers between 1 and 81 excluding the specified numbers
unique_numbers = random.sample([num for num in range(1, 81) if num not in exclude], 75)

#use_fn_view = ext.construct_all_galaxies_leavout("id", unique_numbers)
use_fn_view = ext.construct_all_galaxies_leavout("id",[])
Galaxies = mpc.choose_subset(Galaxies_sorted, use_fn = use_fn_view, cond_fn=cond_fn, comp_use=comp_use)

#Subset to train on (e.g. all galaxies, leavout 5 as validation set):
leavout_idices = [66, 20, 33, 48, 5]
use_fn_train = ext.construct_all_galaxies_leavout("id", leavout_idices)
Galaxies_train = mpc.choose_subset(Galaxies_sorted, use_fn = use_fn_train, cond_fn=cond_fn, comp_use=comp_use)
print("Subset chosen")

latent_space_folder = r"/export/home/ksampath/my_venv/26-07/final_latent_spaces"   #REPLACE WITH PATH TO LATENT SPACE FOLDER
for galaxy_dict in Galaxies_train :
    galaxy_id = galaxy_dict['galaxy']['NIHAO_id']  
    
    latent_space_file = os.path.join(latent_space_folder, f'{galaxy_id}_latent.npy')
    latent_space = list(np.load(latent_space_file))

    for i in range(0,16):
        galaxy_dict['parameters'][f'lv{i+1}'] = latent_space[i]
print("Latent variables appended")

#Data to Flow
#The flow should be trained in normalized coordinates
#Also we want e.g. total stellar mass to be learned in log
#M_stars should be learned in log
LOG_LEARN = ["M_stars"]
#Define the transformations to be used
transformations = (np.log10, )
#Define the manes affected by the transformations (i.e. components of trf_names[i] are supplied to transformations[i])
trf_names = (LOG_LEARN, )
#Define the inverse transformations (these are applied to the model output)
transformations_inv = (lambda x: 10**x, )
#Define the logdets of the transformations needed if the pdf is to be computed
logdets = (ext.logdet_log10,)

Data_flow = mpc.Data_to_flow(mpc.diststack(Galaxies_train), transformations, trf_names, transformations_inv, transformation_logdets=logdets)
print("Data converted to Flow input")

# Separate components and conditions
components = Data_flow[['x', 'y', 'z', 'vx', 'vy', 'vz', 'Z', 'feh', 'ofe', 'age']].values
conditions = Data_flow[['M_stars', 'tau50', 'tau10', 'Z_av', 'lv1', 'lv2', 'lv3', 'lv4', 'lv5', 'lv6', 'lv7', 'lv8', 'lv9', 'lv10', 'lv11', 'lv12', 'lv13', 'lv14', 'lv15', 'lv16']].values

# Convert to PyTorch tensors
components = torch.tensor(components, dtype=torch.float32).to(device)
conditions = torch.tensor(conditions, dtype=torch.float32).to(device)

# Create a dataset and data loader
dataset = TensorDataset(components, conditions)
data_loader = DataLoader(dataset, batch_size=1024, shuffle=True)

save_dir = 'model_checkpoints_cn7'
os.makedirs(save_dir, exist_ok=True)

def save_checkpoint(epoch, step, model, optimizer, train_loss):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss
    }
    torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch}_step_{step}.pth'))

def save_losses(epoch, step, train_loss):
    with open(os.path.join(save_dir, 'losses.txt'), 'a') as f:
        f.write(f"Epoch: {epoch}, Step: {step}, Train Loss: {train_loss}\n")

# Define the model
features = components.shape[1]  # Number of features in components
context = conditions.shape[1]  # Number of features in conditions
transforms = 30  # Number of autoregressive transformations

flow = zuko.flows.coupling.NICE(
    features=features, 
    context=context, 
    transforms=transforms, 
    hidden_features=(128, 128)
).to(device)

# Training setup
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)
lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
num_epochs = 200
step = 0

for epoch in range(num_epochs):
    losses = []
    flow.train()
    train_loss = 0
    for batch in data_loader:
        components_batch, conditions_batch = batch
        # Compute loss
        loss = -flow(conditions_batch).log_prob(components_batch).mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.detach().cpu().item())
        step += 1

        if step % 400 == 0:
            avg_loss = statistics.mean(losses)
            print(f'Epoch {epoch}, Step {step}, Avg Loss: {avg_loss}')

        if step % 3000 == 0:
            save_checkpoint(epoch, step, flow, optimizer, statistics.mean(losses))
            save_losses(epoch, step, statistics.mean(losses))

        #Decrease learning rate every 10 steps until it is smaller than 3*10**-6, then every 120 steps
        if lr_schedule.get_last_lr()[0] <= 3*10**-6:
            decrease_step = 120
        else:
            decrease_step = 10

        #Update learning rate every decrease_step steps
        if step % decrease_step == 0:
            lr_schedule.step()
    
    losses = torch.tensor(losses)
    print(f'({epoch})', losses.mean().item(), '±', losses.std().item())

"""
condition_example = conditions[0:1].unsqueeze(dim=-1)  # Add extra dimension for context
samples = flow.inverse(flow.base_distribution.sample((1000,)).to(device), context=condition_example)
print(samples)"""




