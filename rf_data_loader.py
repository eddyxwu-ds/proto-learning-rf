import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import struct

def read_binary_tim_file(filepath):
    """
    Read a .tim binary file containing interleaved real and imaginary parts.
    Returns complex numpy array.
    """
    with open(filepath, 'rb') as f:
        # Read all bytes
        data = f.read()
    
    # Each sample is 8 bytes (4 bytes real + 4 bytes imaginary)
    num_samples = len(data) // 8
    
    # Unpack the binary data
    samples = struct.unpack('f' * (num_samples * 2), data)
    
    # Convert to complex array
    complex_samples = np.array(samples[::2]) + 1j * np.array(samples[1::2])
    
    return complex_samples

def load_signal_labels(label_file_path):
    """
    Load signal labels from signal_record file.
    Returns dictionary mapping signal index to (modulation_type, parameters).
    """
    labels = {}
    modulation_to_idx = {
        'bpsk': 0, 'qpsk': 1, '8psk': 2, 'dqpsk': 3,
        '16qam': 4, '64qam': 5, '256qam': 6, 'msk': 7
    }
    
    with open(label_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                signal_idx = int(parts[0])
                modulation = parts[1].lower()
                if modulation in modulation_to_idx:
                    labels[signal_idx] = (modulation_to_idx[modulation], parts[2:])
    
    return labels

class RFDataset(Dataset):
    def __init__(self, data_dir, label_file_path, transform=None):
        """
        RF Signal Dataset for prototype learning.
        
        Args:
            data_dir: Directory containing .tim files
            label_file_path: Path to signal_record file
            transform: Optional transform to apply to signals
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Load labels
        self.labels = load_signal_labels(label_file_path)
        
        # Get list of available signal files
        self.signal_files = []
        for filename in os.listdir(data_dir):
            if filename.endswith('.tim'):
                signal_idx = int(filename.split('_')[1].split('.')[0])
                if signal_idx in self.labels:
                    self.signal_files.append((signal_idx, filename))
        
        # Sort by signal index
        self.signal_files.sort(key=lambda x: x[0])
        
        print(f"Loaded {len(self.signal_files)} signals from {data_dir}")
    
    def __len__(self):
        return len(self.signal_files)
    
    def __getitem__(self, idx):
        signal_idx, filename = self.signal_files[idx]
        filepath = os.path.join(self.data_dir, filename)
        
        # Read complex signal
        complex_signal = read_binary_tim_file(filepath)
        
        # Convert to real-valued features (magnitude and phase)
        magnitude = np.abs(complex_signal)
        phase = np.angle(complex_signal)
        
        # Stack magnitude and phase as 2-channel input
        # Normalize to [0, 1] range
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        phase = (phase + np.pi) / (2 * np.pi)  # Normalize phase to [0, 1]
        
        # Stack as 2D array: [2, signal_length]
        signal_features = np.stack([magnitude, phase], axis=0)
        
        # Get label
        label = self.labels[signal_idx][0]
        
        # Convert to torch tensors
        signal_tensor = torch.from_numpy(signal_features).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        if self.transform:
            signal_tensor = self.transform(signal_tensor)
        
        return signal_tensor, label_tensor

def create_rf_dataloaders(data_dir, label_file_path, batch_size=32, train_split=0.8):
    """
    Create train and test dataloaders for RF signals.
    """
    # Create full dataset
    full_dataset = RFDataset(data_dir, label_file_path)
    
    # Split into train and test
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    test_size = dataset_size - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    return train_loader, test_loader, len(train_dataset), len(test_dataset) 