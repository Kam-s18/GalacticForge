# GalacticForge
GalacticForge aims to learn extended galactic distributions of galaxies conditioned on a set of parameters. The pipeline consists of an Autoencoder trained on galaxy images to capture global properties in a latent space, which conditions a Conditional Normalizing Flow model. The conditional normlaizing flow once trained, can be used to sample as many number of stars as required within a galaxy upon providing the conditions.

## Dataset
The dataset consists of high-definition galaxy images (which will be used to train an AutoEncoder in an attempt to learn some global properties of the galaxy) and their corresponsing astrophysical data - Position, Velocity, Abundance, Mass and Age (which will be used to train the conditional normalizng flows, conditioned on the latent space from the AutoEncoder). The galaxy images and their corresponding astrophysical data (in .NPY format) can be found here : [Link to dataset]()

## Preperation of Data
The data cleaning and preperation process followed here is entirely inspired from [GalacticFlow](https://github.com/luwo9/GalacticFlow). From the NIHAO simulation suite, a 10D joint distribution of stellar position (x,y,z), velocity (vx,vy,vz), total metallicity (Z), iron and oxygen abundance ([Fe/H],[O/Fe]) and stellar age are obtained. As paramters/conditions, we also obtain the total stellar mass, median stellar age, the 10-th percentile stellar age and the average stellar metallicity for each galaxy. The data is further cleaned by removing outliers in terms of position and then, metallicity. In this endeavour, certain galaxies are removed from the dataset (both images and astrophysical data). 

## Pipeline
The pipeline consists of an AutoEncoder and a Conditional Normalazing Flow
### AutoEncoder
The AutoEncoder is used to learn global properties from galaxy images. 

The encoder is a series of convolutional layers that progressively reduces the spatial dimensions from the input (3x448x448). The latent space has a dimension of 16. The decoder is a series of transposed convolutional layers that progressively increase the spatial dimension of the latent space back to the original image dimension. Note that the decoder is irrelevant to the GalacticForge. It only exists to test the performance of the AutoEncoder.

The latent space must be able to capture certain global galaxy properties like shape, colour, size, orientation, etc. However, the latent space need not be physically interpretable. Here is how the reconstruction image of a galaxy changes on changing the latent variable values:

These latent variables are provided as conditions along with the parameters from the data preperation stage, to the Conditional Normalizing Flow model. 

### Conditional Normalizing Flows
Conditional Normalizing Flow is used to learn the distribution of stars in a galaxy, conditioned on the latent variables and parameters from the data preperation stage. Various Conditional Normalizing Flow models have been trained and tested out:
a) CNF as provided in [GalacticFlow](https://github.com/luwo9/GalacticFlow):
b) Sum-of-Squares Polynomial Flow (Implement using Zuko library)
c) NICE Flow (Implement using Zuko library)
d) Auto Regressive Flow (Implement using Zuko library)

The results and code of all the above normalizing flows can be found in the repo.





