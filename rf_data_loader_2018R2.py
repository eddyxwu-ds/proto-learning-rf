import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import struct
import glob

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

def load_signal_labels_2018R2(label_file_path):
    """
    Load signal labels from CSPB.ML.2018R2 signal_record file.
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

def find_signal_files(base_dir, pattern="Batch_Dir_*"):
    """
    Find all .tim files across multiple batch directories.
    Returns list of (signal_idx, filepath) tuples.
    """
    signal_files = []
    
    # Find all batch directories
    batch_dirs = glob.glob(os.path.join(base_dir, pattern))
    batch_dirs.sort()  # Ensure consistent ordering
    
    for batch_dir in batch_dirs:
        if os.path.isdir(batch_dir):
            # Find all .tim files in this batch
            tim_files = glob.glob(os.path.join(batch_dir, "signal_*.tim"))
            
            for filepath in tim_files:
                # Extract signal index from filename
                filename = os.path.basename(filepath)
                try:
                    signal_idx = int(filename.split('_')[1].split('.')[0])
                    signal_files.append((signal_idx, filepath))
                except (IndexError, ValueError):
                    print(f"Warning: Could not parse signal index from {filename}")
                    continue
    
    # Sort by signal index
    signal_files.sort(key=lambda x: x[0])
    
    return signal_files

class RFDataset2018R2(Dataset):
    def __init__(self, base_dir, label_file_path, transform=None, max_signals=None):
        """
        RF Signal Dataset for CSPB.ML.2018R2 with prototype learning.
        
        Args:
            base_dir: Base directory containing Batch_Dir_* folders
            label_file_path: Path to signal_record file
            transform: Optional transform to apply to signals
            max_signals: Maximum number of signals to load (for testing)
        """
        self.base_dir = base_dir
        self.transform = transform
        
        # Load labels
        print(f"Loading labels from {label_file_path}...")
        self.labels = load_signal_labels_2018R2(label_file_path)
        print(f"Loaded {len(self.labels)} labels")
        
        # Find all signal files
        print(f"Searching for signal files in {base_dir}...")
        all_signal_files = find_signal_files(base_dir)
        print(f"Found {len(all_signal_files)} signal files")
        
        # Filter to only include signals that have labels
        self.signal_files = []
        for signal_idx, filepath in all_signal_files:
            if signal_idx in self.labels:
                self.signal_files.append((signal_idx, filepath))
        
        # Limit number of signals if specified
        if max_signals is not None:
            self.signal_files = self.signal_files[:max_signals]
        
        print(f"Final dataset: {len(self.signal_files)} signals with labels")
        
        # Verify we have signals for all modulation types
        mod_counts = {}
        for signal_idx, _ in self.signal_files:
            mod_type = self.labels[signal_idx][0]
            mod_counts[mod_type] = mod_counts.get(mod_type, 0) + 1
        
        print("Modulation type distribution:")
        mod_names = ['BPSK', 'QPSK', '8PSK', 'π/4-DQPSK', '16QAM', '64QAM', '256QAM', 'MSK']
        for i, name in enumerate(mod_names):
            count = mod_counts.get(i, 0)
            print(f"  {name}: {count}")
    
    def __len__(self):
        return len(self.signal_files)
    
    def __getitem__(self, idx):
        signal_idx, filepath = self.signal_files[idx]
        
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

def create_rf_dataloaders_2018R2(base_dir, label_file_path, batch_size=32, train_split=0.8, max_signals=None):
    """
    Create train and test dataloaders for CSPB.ML.2018R2 RF signals.
    """
    # Create full dataset
    full_dataset = RFDataset2018R2(base_dir, label_file_path, max_signals=max_signals)
    
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