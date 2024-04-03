'''
Going to write scripts
'''

import torch
import h5py as h5

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


class weather_dataset(torch.utils.data.Dataset):
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


def get_dataloaders(fname:str, batch_size:int, split:int) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return train_loader, test_loader