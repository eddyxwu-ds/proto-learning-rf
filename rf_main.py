import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse
import os
import numpy as np
from rf_data_loader import create_rf_dataloaders
from rf_models import RFNet, regularization
from train_utils import lr_scheduler

def train_rf_model(model, optimizer, lrate, num_epochs, reg, train_loader, test_loader, 
                   dataset_train_len, dataset_test_len, plotsFileName, csvFileName):
    """
    Training function adapted for RF signals.
    """
    epochs = []
    train_acc = []
    test_acc = []
    train_loss = []
    test_loss = []
    train_error = []
    test_error = []
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        epochs.append(epoch)
        optimizer = lr_scheduler(optimizer, lrate, epoch)
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('*' * 70)
        
        running_loss = 0.0
        running_corrects = 0.0
        train_batch_ctr = 0.0
        
        for i, (signal, label) in enumerate(train_loader):
            signal, label = torch.autograd.Variable(signal, requires_grad=True), torch.autograd.Variable(label, requires_grad=False)
            
            optimizer.zero_grad()
            
            features, centers, distance, outputs = model(signal)
            _, preds = torch.max(distance, 1)
            
            loss1 = F.nll_loss(outputs, label)
            loss2 = regularization(features, centers, label)
            loss = loss1 + reg * loss2
            
            loss.backward()
            optimizer.step()
            train_batch_ctr = train_batch_ctr + 1
            
            running_loss += loss.item()
            running_corrects += torch.sum(preds == label.data)
            
            epoch_acc = (float(running_corrects) / float(dataset_train_len))
        
        print('Train corrects: {} Train samples: {} Train accuracy: {}'.format(
            running_corrects, dataset_train_len, epoch_acc))
        train_acc.append(epoch_acc)
        train_loss.append(1.0 * running_loss / train_batch_ctr)
        train_error.append(((dataset_train_len) - running_corrects) / (dataset_train_len))
        
        # Test phase
        model.eval()
        test_running_corrects = 0.0
        test_batch_ctr = 0.0
        test_running_loss = 0.0
        
        with torch.no_grad():
            for signal, label in test_loader:
                signal, label = torch.autograd.Variable(signal), torch.autograd.Variable(label)
                
                features, centers, distance, test_outputs = model(signal)
                _, predicted_test = torch.max(distance, 1)
                
                loss1 = F.nll_loss(test_outputs, label)
                loss2 = regularization(features, centers, label)
                loss = loss1 + reg * loss2
                
                test_running_loss += loss.item()
                test_batch_ctr = test_batch_ctr + 1
                
                test_running_corrects += torch.sum(predicted_test == label.data)
                test_epoch_acc = (float(test_running_corrects) / float(dataset_test_len))
        
        if test_epoch_acc > best_acc:
            torch.save(model, 'best_rf_model.pt')
            best_acc = test_epoch_acc
        
        test_acc.append(test_epoch_acc)
        test_loss.append(1.0 * test_running_loss / test_batch_ctr)
        
        print('Test corrects: {} Test samples: {} Test accuracy {}'.format(
            test_running_corrects, dataset_test_len, test_epoch_acc))
        print('Train loss: {} Test loss: {}'.format(train_loss[epoch], test_loss[epoch]))
        print('*' * 70)
    
    torch.save(model, 'final_rf_model.pt')
    return model, train_acc, test_acc, train_loss, test_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01, help='initial_learning_rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_classes', type=int, default=8, help='number of modulation classes')
    parser.add_argument('--h', type=int, default=2, help='dimension of the hidden layer')
    parser.add_argument('--scale', type=float, default=2, help='scaling factor for distance')
    parser.add_argument('--reg', type=float, default=0.001, help='regularization coefficient')
    parser.add_argument('--data_dir', type=str, default='Batch_Dir_1', help='directory containing .tim files')
    parser.add_argument('--label_file', type=str, default='signal_record_first_20000.txt', help='path to label file')
    
    args, _ = parser.parse_known_args()
    
    # Check if label file exists
    if not os.path.exists(args.label_file):
        print(f"Warning: Label file {args.label_file} not found.")
        print("You need to download the signal_record_first_20000.txt file from the CSP Blog.")
        print("For now, we'll create a dummy label file for testing...")
        
        # Create a dummy label file for testing
        with open(args.label_file, 'w') as f:
            for i in range(1, 4001):  # Assuming 4000 signals in Batch 1
                # Randomly assign one of 8 modulation types
                mod_types = ['bpsk', 'qpsk', '8psk', 'dqpsk', '16qam', '64qam', '256qam', 'msk']
                mod_type = mod_types[i % 8]
                f.write(f"{i} {mod_type} 11 0.0 0.5 10 9 10.0 0.0\n")
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, test_loader, dataset_train_len, dataset_test_len = create_rf_dataloaders(
        args.data_dir, args.label_file, batch_size=args.batch_size
    )
    
    # Create model
    model = RFNet(args.h, args.num_classes, args.scale)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(model)
    
    # Setup optimizer
    lrate = args.lr
    optimizer = optim.SGD(model.parameters(), lr=lrate, momentum=0.9, weight_decay=1e-4)
    
    # Training parameters
    num_epochs = 30
    
    # File names for saving results
    plotsFileName = './plots/rf_signals'
    csvFileName = './stats/rf_log.csv'
    
    # Create directories if they don't exist
    os.makedirs('./plots', exist_ok=True)
    os.makedirs('./stats', exist_ok=True)
    
    print("Starting training...")
    print(f"Training on {dataset_train_len} samples, testing on {dataset_test_len} samples")
    
    # Train the model
    model, train_acc, test_acc, train_loss, test_loss = train_rf_model(
        model, optimizer, lrate, num_epochs, args.reg, 
        train_loader, test_loader, dataset_train_len, dataset_test_len,
        plotsFileName, csvFileName
    )
    
    print("Training completed!")
    print(f"Best test accuracy: {max(test_acc):.4f}") 