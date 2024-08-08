#Import libraires
import processing
import externalize as ext
import flowcode2 as flowcode
import res_flow_vis as visual
import final_AE

import torch
import pandas as pd
import numpy as np
import os
import random
import time

#Filename for a given run
filename = "leavout_MttZ_all_CL2_24_10_512_8_lo[66, 20, 33, 48, 5]" #"leavout_MttZ_MWs_CL2_24_10_512_8_lo5"

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

#Choose deviceuda"
device = torch.device("cuda")

#Hyperparameters of the flow
LAYER_TYPE = flowcode.NSF_CL2
N_LAYERS = 16
COND_NAMES = mpc.cond_names["galaxy"]
DIM_COND = len(COND_NAMES)
DIM_NOTCOND = 10
SPLIT = 0.5
K = 10
B = 3
BASE_NETWORK = flowcode.MLP
BASE_NETWORK_N_LAYERS = 4
BASE_NETWORK_N_HIDDEN = 128
BASE_NETWORK_LEAKY_RELU_SLOPE = 0.2

SPLIT = {"split":SPLIT} if LAYER_TYPE == flowcode.NSF_CL else {}

#Training hyperparameters
N_EPOCHS = 12
INIT_LR = 0.00009
GAMMA = 0.998
BATCH_SIZE = 1024

#Instantiate the model
model = flowcode.NSFlow(N_LAYERS, DIM_NOTCOND, DIM_COND, LAYER_TYPE, **SPLIT, K=K, B=B, network=BASE_NETWORK, network_args=(BASE_NETWORK_N_HIDDEN,BASE_NETWORK_N_LAYERS,BASE_NETWORK_LEAKY_RELU_SLOPE))
model = model.to(device)

print("Training starts")

train_loss_saver = []
start = time.perf_counter()
flowcode.train_flow(model, Data_flow, COND_NAMES, N_EPOCHS, lr=INIT_LR, batch_size=BATCH_SIZE, loss_saver=train_loss_saver, gamma=GAMMA)
end = time.perf_counter()
torch.save(model.state_dict(), f"saves/{filename}.pth")
np.save(f"saves/loss_{filename}.npy",np.array(train_loss_saver+[end-start]))