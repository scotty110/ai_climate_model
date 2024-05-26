'''
Utility file, holds data prep, models, and eval functions.
Training loop is in main.py
'''
import torch
import h5py as h5

import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

#-------------------------

'''
Data loading and processing functions
'''

def load_hdf5(filename:str):
    '''
    Load data from an HDF5 file and return a list of dictionaries.
    Inputs:
        - filename (str): Path to the HDF5 file.
    Outputs:
        - data (list): List of dictionaries, where each dictionary represents an entry in the original list.
    '''
    data = []  # List to hold dictionaries
    with h5.File(filename, 'r') as f:
        # Iterate through groups (each representing an entry in the original list)
        for group_name in f:
            group = f[group_name]
            # Reconstruct dictionary from datasets and attributes
            entry = {
                # Attributes (metadata)
                'day': group.attrs['day'],
                'region': group.attrs['region'],
                'time': group.attrs['time'],

                # groups (numpy arrays)
                'landmass': group['landmass'][...],  # Use [...] to read the full dataset
                'x': group['x'][...],
                'y': group['y'][...],
            }
            data.append(entry)
    return data


def stack_data(data:list[dict], key:str) -> torch.Tensor:
    return torch.stack([torch.tensor(entry[key]) for entry in data])


def generate_stacks(data:list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Create a PyTorch DataLoader from the data.
    Inputs:
        - data (list): List of dictionaries, where each dictionary represents an entry in the original list.
    Outputs:
        - landmass (torch.Tensor): Tensor of landmass data.
        - x (torch.Tensor): Tensor of x-coordinate data.
        - y (torch.Tensor): Tensor of y-coordinate data.
    '''
    landmass = stack_data(data, 'landmass')

    x = stack_data(data, 'x')
    x = x.transpose(2, 1)

    y = stack_data(data, 'y')
    y = y.transpose(2, 1)
    
    return (landmass, x, y)


class weather_dataset(Dataset):
    '''
    PyTorch Dataset class for weather data.
    '''
    def __init__(self, data:list[dict]):
        self.landmass, self.x, self.y = generate_stacks(data)
        self.length = len(self.landmass)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (self.landmass[idx], self.x[idx], self.y[idx])


def get_dataloaders(fname:str, batch_size:int, split:int) -> tuple[DataLoader, DataLoader]:
    '''
    Create PyTorch DataLoader objects for training and validation data.
    Inputs:
        - fname (str): Path to the HDF5 file.
        - batch_size (int): Batch size for the DataLoader objects.
        - split (float): Fraction of the data to use for training.     
    Outputs:
        - train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        - test_loader (torch.utils.data.DataLoader): DataLoader for test data.
    '''
    # Load data and create tensor 
    data = load_hdf5(fname)
    dataset = weather_dataset(data)
    
    train_size = int(split * len(dataset))
    val_size = len(dataset) - train_size
    
    # Split data into training and validation sets
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    return train_loader, test_loader


#-------------------------

'''
Models
'''

class DepthWiseConv2d(nn.Module):
    def __init__(self, out_dim:int, input_shape:list[int]=[70,3,2,2]):
        super(DepthWiseConv2d, self).__init__()
        '''
        Want to convolve over the 2x2 grid of the input values, then apply a pointwise convolution over the features
        May want to go over again with fresh eyes, Dealing with 5D tensor is a bit confusing
        Args:
            - output_dim (int): Number of output features
            - input_shape (list): Shape of the input tensor, default is [70,3,2,2] since maybe want to change data shape in future???
        '''
        # Input: [batch_size, 70*3, 2, 2]
        self.input_shape = input_shape
        self.depthwise = nn.Conv2d(
                            in_channels=self.input_shape[0]*self.input_shape[1], 
                            out_channels=self.input_shape[0]*self.input_shape[1], 
                            kernel_size=self.input_shape[2:], 
                            groups=self.input_shape[0]*self.input_shape[1])
        self.pointwise = nn.Conv2d(
                            in_channels=self.input_shape[0]*self.input_shape[1], 
                            out_channels=self.input_shape[0], 
                            kernel_size=1)
        
        self.out_dim = out_dim
        self.linear = nn.Linear(input_shape[0], self.input_shape[0]*self.out_dim)

    def forward(self, x:torch.tensor) -> torch.tensor:
        # Do depthwise convolution (no idea if this is a good idea)
        x = x.view(x.size(0), -1, x.size(3), x.size(4))
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = x.squeeze()

        # Want to add an output dim (could handle in larger model, but for now just do it here)
        x = self.linear(x)
        x = x.reshape(x.size(0), self.input_shape[0], self.out_dim)
        return x


class InnerTransformer(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, num_layers:int=2):
        super(InnerTransformer, self).__init__()
        '''
        Args:
            - input_dim (int): Number of input features
            - output_dim (int): Number of output features
            - num_layers (int): Number of transformer layers, default is 1
        '''
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Build transformer layers
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=1) for _ in range(self.num_layers)]) # Make it simple for now

        # Linear layer to output
        self.linear = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x:torch.tensor) -> torch.tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ConvTrans(nn.Module):
    def __init__(self, input_shape:list[int]=[71,3,2,2], inner_shape:int=int(71*2), out_dim:int=int(2*2*2), num_layers:int=2):
        super(ConvTrans, self).__init__()
        '''
        Build a DepthWiseConv2d model followed by a Transformer model followed by an output layer(tbd)
        Args:
            - input_shape (list): Shape of the input tensor, default is [71,3,2,2] (70 cells, 3 features, 2x2 grid, followed by spacial cell). 
            - inner_shape (int): Number of features to pass to the transformer, default is 70*2
            - out_dim (int): Number of output features, default is 70*2*2*2 (70 cells, 2 features, 2x2 grid)
            - num_layers (int): Number of transformer layers, default is 2
        '''
        self.depthwise = DepthWiseConv2d(out_dim=inner_shape, input_shape=input_shape)
        self.transformer = InnerTransformer(input_dim=inner_shape, output_dim=out_dim, num_layers=num_layers)
        self.output_cov = nn.Conv1d(in_channels=input_shape[0], out_channels=70, kernel_size=1)
        self.output_linear = nn.Linear(inner_shape, out_dim)

    def forward(self, x:torch.tensor) -> torch.tensor:
        x = self.depthwise(x)
        x = self.transformer(x)
        x = self.output_cov(x)
        x = x.squeeze()
        x = self.output_linear(x)
        x = x.reshape(x.size(0), 70, 2, 2, 2) # Maybe make args??? IDK
        return x
