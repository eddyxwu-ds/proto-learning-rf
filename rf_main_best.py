import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse
import os
import numpy as np
from rf_data_loader_improved import create_rf_dataloaders_improved
from rf_models_improved import RFNetImproved, regularization
from train_utils import lr_scheduler

def train_rf_model_best(model, optimizer, lrate, num_epochs, reg, train_loader, test_loader, 
                       dataset_train_len, dataset_test_len):
    """
    Best training function with comprehensive monitoring.
    """
    epochs = []
    train_acc = []
    test_acc = []
    train_loss = []
    test_loss = []
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
        
        # Track per-class accuracy
        class_correct = {i: 0 for i in range(8)}
        class_total = {i: 0 for i in range(8)}
        
        for i, (signal, label) in enumerate(train_loader):
            signal, label = torch.autograd.Variable(signal, requires_grad=True), torch.autograd.Variable(label, requires_grad=False)
            
            optimizer.zero_grad()
            
            features, centers, distance, outputs = model(signal)
            _, preds = torch.max(distance, 1)
            
            # Track per-class accuracy
            for j in range(len(label)):
                class_correct[label[j].item()] += (preds[j] == label[j]).item()
                class_total[label[j].item()] += 1
            
            loss1 = F.nll_loss(outputs, label)
            loss2 = regularization(features, centers, label)
            loss = loss1 + reg * loss2
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_batch_ctr = train_batch_ctr + 1
            
            running_loss += loss.item()
            running_corrects += torch.sum(preds == label.data)
            
            epoch_acc = (float(running_corrects) / float(dataset_train_len))
        
        # Print per-class accuracy
        print('Per-class training accuracy:')
        mod_names = ['BPSK', 'QPSK', '8PSK', 'π/4-DQPSK', '16QAM', '64QAM', '256QAM', 'MSK']
        for i in range(8):
            if class_total[i] > 0:
                acc = class_correct[i] / class_total[i]
                print(f'  {mod_names[i]}: {acc:.3f} ({class_correct[i]}/{class_total[i]})')
        
        print('Train corrects: {} Train samples: {} Train accuracy: {}'.format(
            running_corrects, dataset_train_len, epoch_acc))
        train_acc.append(epoch_acc)
        train_loss.append(1.0 * running_loss / train_batch_ctr)
        
        # Test phase
        model.eval()
        test_running_corrects = 0.0
        test_batch_ctr = 0.0
        test_running_loss = 0.0
        
        # Track test per-class accuracy
        test_class_correct = {i: 0 for i in range(8)}
        test_class_total = {i: 0 for i in range(8)}
        
        with torch.no_grad():
            for signal, label in test_loader:
                signal, label = torch.autograd.Variable(signal), torch.autograd.Variable(label)
                
                features, centers, distance, test_outputs = model(signal)
                _, predicted_test = torch.max(distance, 1)
                
                # Track per-class accuracy
                for j in range(len(label)):
                    test_class_correct[label[j].item()] += (predicted_test[j] == label[j]).item()
                    test_class_total[label[j].item()] += 1
                
                loss1 = F.nll_loss(test_outputs, label)
                loss2 = regularization(features, centers, label)
                loss = loss1 + reg * loss2
                
                test_running_loss += loss.item()
                test_batch_ctr = test_batch_ctr + 1
                
                test_running_corrects += torch.sum(predicted_test == label.data)
                test_epoch_acc = (float(test_running_corrects) / float(dataset_test_len))
        
        # Print test per-class accuracy
        print('Per-class test accuracy:')
        for i in range(8):
            if test_class_total[i] > 0:
                acc = test_class_correct[i] / test_class_total[i]
                print(f'  {mod_names[i]}: {acc:.3f} ({test_class_correct[i]}/{test_class_total[i]})')
        
        if test_epoch_acc > best_acc:
            torch.save(model, 'best_rf_model_best.pt')
            best_acc = test_epoch_acc
        
        test_acc.append(test_epoch_acc)
        test_loss.append(1.0 * test_running_loss / test_batch_ctr)
        
        print('Test corrects: {} Test samples: {} Test accuracy {}'.format(
            test_running_corrects, dataset_test_len, test_epoch_acc))
        print('Train loss: {} Test loss: {}'.format(train_loss[epoch], test_loss[epoch]))
        print('*' * 70)
    
    torch.save(model, 'final_rf_model_best.pt')
    return model, train_acc, test_acc, train_loss, test_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0005, help='initial_learning_rate')  # Much lower
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')  # Larger
    parser.add_argument('--num_classes', type=int, default=8, help='number of modulation classes')
    parser.add_argument('--h', type=int, default=2, help='dimension of the hidden layer')
    parser.add_argument('--scale', type=float, default=2, help='scaling factor for distance')
    parser.add_argument('--reg', type=float, default=0.001, help='regularization coefficient')
    parser.add_argument('--data_dir', type=str, default='.', help='base directory containing Batch_Dir_* folders')
    parser.add_argument('--label_file', type=str, default='signal_record_C_2023.txt', help='path to label file')
    parser.add_argument('--max_signals', type=int, default=None, help='maximum number of signals to load (for testing)')
    parser.add_argument('--feature_type', type=str, default='iq', help='feature type: iq, magnitude_phase, spectrogram, iq_normalized')
    
    args, _ = parser.parse_known_args()
    
    # Check if label file exists
    if not os.path.exists(args.label_file):
        print(f"Error: Label file {args.label_file} not found.")
        exit(1)
    
    # Create dataloaders with improved features
    print("Creating improved dataloaders for CSPB.ML.2018R2...")
    train_loader, test_loader, dataset_train_len, dataset_test_len = create_rf_dataloaders_improved(
        args.data_dir, args.label_file, batch_size=args.batch_size, 
        max_signals=args.max_signals, feature_type=args.feature_type
    )
    
    # Create improved model
    model = RFNetImproved(args.h, args.num_classes, args.scale, dropout_rate=0.3)
    print(f"Improved model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Setup optimizer with better parameters
    lrate = args.lr
    optimizer = optim.AdamW(model.parameters(), lr=lrate, weight_decay=1e-4)  # Use AdamW
    
    # Training parameters
    num_epochs = 50  # More epochs
    
    print("Starting best training with real labels...")
    print(f"Training on {dataset_train_len} samples, testing on {dataset_test_len} samples")
    print(f"Learning rate: {lrate}, Batch size: {args.batch_size}")
    print(f"Feature type: {args.feature_type}")
    
    # Train the model
    model, train_acc, test_acc, train_loss, test_loss = train_rf_model_best(
        model, optimizer, lrate, num_epochs, args.reg, 
        train_loader, test_loader, dataset_train_len, dataset_test_len
    )
    
    print("Training completed!")
    print(f"Best test accuracy: {max(test_acc):.4f}")
    
    # Print final results
    print("\nFinal Results:")
    print(f"Best test accuracy: {max(test_acc):.4f}")
    print(f"Final train accuracy: {train_acc[-1]:.4f}")
    print(f"Final test accuracy: {test_acc[-1]:.4f}") 