# GalacticForge

GalacticForge aims to learn extended galactic distributions of galaxies conditioned on a set of parameters. The pipeline consists of an Autoencoder trained on galaxy images to capture global properties in a latent space, which conditions a Conditional Normalizing Flow model. The conditional normlaizing flow once trained, can be used to sample as many number of stars as required within a galaxy upon providing the conditions.
