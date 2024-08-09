# GalacticForge
GalacticForge aims to learn extended galactic distributions of galaxies conditioned on a set of parameters. The pipeline consists of an Autoencoder trained on galaxy images to capture global properties in a latent space, which conditions a Conditional Normalizing Flow model. The conditional normalizing flow once trained, can be used to sample as many number of stars as required within a galaxy upon providing the conditions.

## Dataset
The dataset consists of high-definition galaxy images (which will be used to train an AutoEncoder in an attempt to learn some global properties of the galaxy) and their corresponsing astrophysical data - Position, Velocity, Abundance, Mass and Age (which will be used to train the conditional normalizng flows, conditioned on the latent space from the AutoEncoder). The galaxy images and their corresponding astrophysical data (in .NPY format) can be found here : [Link to dataset](https://drive.google.com/drive/folders/1hidwFz0X_smICtTIy0VJhrkZ-21zxRVd?usp=sharing)

## Preparation of Data
The data cleaning and preparation process followed here is inspired from [GalacticFlow](https://github.com/luwo9/GalacticFlow). From the NIHAO simulation suite, a 10D joint distribution of stellar position (x,y,z), velocity (vx,vy,vz), total metallicity (Z), iron and oxygen abundance ([Fe/H],[O/Fe]) and stellar age are obtained. As paramters/conditions, we also obtain the total stellar mass, median stellar age, the 10-th percentile stellar age and the average stellar metallicity for each galaxy. The data is further cleaned by removing outliers in terms of position and then, metallicity. In this endeavour, certain galaxies are removed from the dataset (both from images and astrophysical data). 

## Pipeline
The pipeline consists of an AutoEncoder and a Conditional Normalazing Flow
### AutoEncoder
The AutoEncoder is used to learn global properties from galaxy images. 

The encoder is a series of convolutional layers that progressively reduces the spatial dimensions from the input (3x448x448). The latent space has a dimension of 16. The decoder is a series of transposed convolutional layers that progressively increase the spatial dimension of the latent space back to the original image dimension. Note that the decoder is irrelevant to GalacticForge. It only exists to test the performance of the AutoEncoder.

The latent space must be able to capture certain global galaxy properties like shape, colour, size, orientation, etc. However, the latent space need not be physically interpretable. Here is how the reconstruction image of a galaxy changes on changing the latent variable values:
![output6_deep](https://github.com/user-attachments/assets/43f8eb29-9e5d-44f5-a868-c5f14b84a051)

These latent variables are provided as conditions along with the parameters from the data preparation stage, to the Conditional Normalizing Flow model. 

### Conditional Normalizing Flows
Conditional Normalizing Flow is used to learn the distribution of stars in a galaxy, conditioned on the latent variables and parameters from the data preperation stage. Various Conditional Normalizing Flow models have been trained and tested out: <br />
a) CNF as provided in [GalacticFlow](https://github.com/luwo9/GalacticFlow) <br />
b) Sum-of-Squares Polynomial Flow (Implement using Zuko library) <br />
c) NICE Flow (Implement using Zuko library) <br />
d) Auto Regressive Flow (Implement using Zuko library) <br />

The results and code of all the above normalizing flows can be found in the repo.

## Repo Structure and File Information
1) The `AutoEncoder` folder contains the notebook with the code for the Autoencoder, latent space for each galaxy, weights of the final AutoEncoder after training saved in a .PTH file and some images representing how the AutoEncoder performs.
2) The folders `Autoregressive_flow_model`, `Main_flow_model`, `Main_flow_model_hypertuned`, `NICE_flow_model` and `SOSPF_flow_model` contains the implementation of various normalizing flows, organized into three sub-folders - `Codes`, `Model Checkpoint and Losses` and `Outputs`.
3) In each of the above models folders, the `Codes` sub-folder contains - `externalize.py`, `flowcode.py`, `processing.py`, and `res_flow_vis.py` which are dependencies required for GalacticForge and are adapted from [GalacticFlow](https://github.com/luwo9/GalacticFlow), with a few changes made. `Final_AE.py` is the file that contains the code for the AutoEncoder. `workflow.py` contains the code that binds all other dependecies and implements the following: Cleaning of data, Optionally train the AE, Choose subsets of data, Process data, define model architecture, train the model and save checkpoints of the model during training.
4) In each of the above model folders, the `Outputs` sub-folder contains the output screenshots for the model (indicating its performance).
5) In each of the above model folders, the `Model Checkpoints and Losses` folder contains the weights of the final model after training saved in .PTH file.
6) The `Main_Workflow_Galactic_Flow_model.ipynb` contains the overall workflow from data cleaning to model training till sampling and inference for a CNF architecture as in [GalacticFlow](https://github.com/luwo9/GalacticFlow)
7) The `Main_Workflow_custom_model.ipynb` contains the overall workflow from data cleaning to model training till sampling and inference for a CNF architecture defined from the [zuko]([https://github.com/luwo9/GalacticFlow](https://zuko.readthedocs.io/stable/index.html)) library. This can be an AutoRegressive CNF or NICE CNF or SOSPF CNF model.

## How to Use?
1) The notebooks `Main_Workflow_Galactic_Flow_model.ipynb` and `Main_Workflow_custom_model.ipynb` can be run entirely and that would Clean the Data, Train Autoencoder, Train your conditional normalizing flow and perform sampling and inference.
2) However, [Link](https://drive.google.com/drive/folders/1hidwFz0X_smICtTIy0VJhrkZ-21zxRVd?usp=sharing) conains the latent space for all the galaxy images considered, thereby eliminating the need to train the autoencoder from scratch unless required.
3) Training of conditional normalizing flow take >24 hours and hence it is recommended to run the `workflow.py` script on GPU rather than running it in the notebook. This would create a folder at specified location with checkpoints and losses during training.
4) One can also load the checkpoint (provided in the `Model Checkpoints and Losses` folder) in order to bypass the training and directly proceed to inference.
5) The section where inference and sampling starts is indicated in the notebooks `Main_Workflow_Galactic_Flow_model.ipynb` and `Main_Workflow_custom_model.ipynb`. The entire notebook would still have to be run (skip any training part and instead load checkpoints and latent variables) for sampling and inference.

## Some Figures
1) Performance of CNF Model as provided in [GalacticFlow](https://github.com/luwo9/GalacticFlow) with hyperparameter tuning:
   ![output1](https://github.com/user-attachments/assets/1b595b0f-baf9-49df-a6db-74742ce7f68c)

2) Performance of SOSPF Model:
   ![outputa4](https://github.com/user-attachments/assets/8173ec6b-6bcc-447c-82b3-b9ae99b55a3c)

