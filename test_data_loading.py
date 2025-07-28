import torch
import numpy as np
from rf_data_loader_2018R2 import create_rf_dataloaders_2018R2
import matplotlib.pyplot as plt

# Test data loading
print("Testing data loading...")
train_loader, test_loader, train_len, test_len = create_rf_dataloaders_2018R2(
    '.', 'signal_record_C_2023.txt', max_signals=100, batch_size=10
)

print(f"Train loader: {train_len} samples")
print(f"Test loader: {test_len} samples")

# Get a batch and examine it
for batch_idx, (signals, labels) in enumerate(train_loader):
    print(f"\nBatch {batch_idx}:")
    print(f"Signals shape: {signals.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels: {labels}")
    
    # Check signal statistics
    print(f"Signal min: {signals.min():.4f}")
    print(f"Signal max: {signals.max():.4f}")
    print(f"Signal mean: {signals.mean():.4f}")
    print(f"Signal std: {signals.std():.4f}")
    
    # Plot first signal
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(signals[0, 0, :1000])  # Magnitude channel
    plt.title(f'Signal 0 - Magnitude (Label: {labels[0]})')
    plt.xlabel('Sample')
    plt.ylabel('Magnitude')
    
    plt.subplot(1, 2, 2)
    plt.plot(signals[0, 1, :1000])  # Phase channel
    plt.title(f'Signal 0 - Phase (Label: {labels[0]})')
    plt.xlabel('Sample')
    plt.ylabel('Phase')
    
    plt.tight_layout()
    plt.savefig('test_signal.png')
    plt.show()
    
    break  # Only examine first batch

print("\nData loading test completed!") 