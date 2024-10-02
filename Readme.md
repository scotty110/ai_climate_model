# AI Models for Hybrid CAM 
This repo contains the torch models for running hybrid CAM simulations. 

## Produce Data NetCDF
Read in netcdf and convert to hdf5 dataset. 
-  Run `process_data.py` to generate the data. 
-  Run the `partition.ipynb` notebook to partition the data into train and test splits

## Simple Noise
This is a simple drop in model that perturbes the temperature by 1\% of the mean.

## Transormer Model
We decided to use a tranformer architeture due to its ability to predict values based on a series of inputs. Traditionally transformers "mix" the tokens then predict the next token. In our case we are mixing the values from the various cell levels and trying to predict the standard deviation for a gaussian distribution. This mixing means the model can learn dependencies that are not immediately clear.

This contains 2 notebooks.  
- 2x2 which preserves spatial information of the 2x2 grid or cells. This uses a "Convolutional Transformer", which does a depthwise convolution over the spacial cells and then feeds the output into a transformer, the idea was to incorporate spacial information to increase accuracy. 
- 1x1 which treats the grid of cells independent and flattens them so all spacial information is lost. This is just a normal transformer. 
