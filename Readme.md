# AI Models for Hybrid CAM 
This repo contains the torch models for running hybrid CAM simulations. 

## Produce Data NetCDF
Read in netcdf and convert to hdf5 dataset. 
-  Run `process_data.py` to generate the data. 
-  Run the `partition.ipynb` notebook to partition the data into train and test splits

## Simple Noise
This is a simple drop in model that perturbes the temperature by 1\% of the mean.

## Transormer Model
This contains 2 notebooks. 
- 2x2 which preserves spatial information of the 2x2 grid or cells
- 1x1 which treats the grid of cells independent and flattens them
