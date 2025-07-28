import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from rf_data_loader import RFDataset
from rf_models import RFNet
from torch.utils.data import DataLoader

def extract_rf_features(model, data_loader, device='cpu'):
    """
    Extract features from trained RF model.
    """
    model.eval()
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for signal, label in data_loader:
            signal = signal.to(device)
            label = label.to(device)
            
            # Get features (before DCE loss)
            features, centers, distance, outputs = model(signal)
            
            features_list.append(features.cpu().numpy())
            labels_list.append(label.cpu().numpy())
    
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    return features, labels

def visualize_rf_prototypes(model, data_loader, save_path='rf_prototypes.png'):
    """
    Visualize RF signal prototypes in 2D feature space.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Extract features
    features, labels = extract_rf_features(model, data_loader, device)
    
    # Get prototype centers
    centers = model.dce.centers.cpu().detach().numpy()
    
    # Modulation type names
    mod_names = ['BPSK', 'QPSK', '8PSK', 'π/4-DQPSK', '16QAM', '64QAM', '256QAM', 'MSK']
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Create scatter plot
    plt.figure(figsize=(12, 10))
    
    # Plot features
    for i in range(8):
        mask = labels == i
        plt.scatter(features[mask, 0], features[mask, 1], 
                   c=colors[i], label=mod_names[i], alpha=0.6, s=20)
    
    # Plot prototype centers
    for i in range(8):
        plt.scatter(centers[0, i], centers[1, i], 
                   c=colors[i], marker='*', s=300, edgecolors='black', linewidth=2,
                   label=f'{mod_names[i]} Prototype')
    
    plt.xlabel('Feature Dimension 1')
    plt.ylabel('Feature Dimension 2')
    plt.title('RF Signal Prototype Learning Visualization')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Prototype visualization saved to {save_path}")
    
    return features, labels, centers

def plot_signal_examples(data_loader, num_examples=8, save_path='rf_signal_examples.png'):
    """
    Plot example RF signals from different modulation types.
    """
    mod_names = ['BPSK', 'QPSK', '8PSK', 'π/4-DQPSK', '16QAM', '64QAM', '256QAM', 'MSK']
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # Get one example from each class
    examples_per_class = {}
    
    for signal, label in data_loader:
        for i in range(signal.shape[0]):
            class_idx = label[i].item()
            if class_idx not in examples_per_class:
                examples_per_class[class_idx] = signal[i]
                if len(examples_per_class) == 8:
                    break
        if len(examples_per_class) == 8:
            break
    
    # Plot examples
    for i in range(8):
        if i in examples_per_class:
            signal_data = examples_per_class[i]
            
            # Plot magnitude and phase
            magnitude = signal_data[0, :1000]  # First 1000 samples
            phase = signal_data[1, :1000]
            
            axes[i].plot(magnitude, label='Magnitude', alpha=0.7)
            axes[i].plot(phase, label='Phase', alpha=0.7)
            axes[i].set_title(f'{mod_names[i]}')
            axes[i].set_xlabel('Sample')
            axes[i].set_ylabel('Normalized Value')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Signal examples saved to {save_path}")

if __name__ == '__main__':
    # Load trained model
    model_path = 'best_rf_model.pt'
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first.")
        exit(1)
    
    model = torch.load(model_path, map_location='cpu')
    print("Loaded trained model")
    
    # Create dataset and dataloader for visualization
    data_dir = 'Batch_Dir_1'
    label_file = 'signal_record_first_20000.txt'
    
    if not os.path.exists(label_file):
        print(f"Label file {label_file} not found. Please download it from the CSP Blog.")
        exit(1)
    
    dataset = RFDataset(data_dir, label_file)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    
    print(f"Created dataloader with {len(dataset)} samples")
    
    # Visualize prototypes
    print("Creating prototype visualization...")
    features, labels, centers = visualize_rf_prototypes(model, dataloader)
    
    # Plot signal examples
    print("Creating signal examples plot...")
    plot_signal_examples(dataloader)
    
    print("Visualization completed!") 