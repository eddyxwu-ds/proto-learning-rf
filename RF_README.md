# RF Signal Prototype Learning

This project adapts the prototype learning approach from the original MNIST implementation to work with RF (Radio Frequency) signals for modulation classification.

## Overview

The system uses a **Distance-based Cross Entropy (DCE) loss** to learn prototypes for 8 different modulation types:
- BPSK (Binary Phase Shift Keying)
- QPSK (Quadrature Phase Shift Keying)  
- 8PSK (8-Phase Shift Keying)
- π/4-DQPSK (π/4-Differential Quadrature Phase Shift Keying)
- 16QAM (16-Quadrature Amplitude Modulation)
- 64QAM (64-Quadrature Amplitude Modulation)
- 256QAM (256-Quadrature Amplitude Modulation)
- MSK (Minimum Shift Keying)

## Dataset

The system is designed to work with the **CSPB.ML.2022R2** dataset from the CSP Blog. This dataset contains:
- Binary `.tim` files with interleaved real and imaginary parts
- Complex RF signals with various modulation types
- Signal parameters including SNR, carrier frequency offset, etc.

## Files

### Core Files
- `rf_data_loader.py` - Data loading and preprocessing for RF signals
- `rf_models.py` - 1D CNN model with DCE loss for RF signals
- `rf_main.py` - Main training script
- `rf_visualize.py` - Visualization tools for prototypes and signals

### Original MNIST Files (for reference)
- `main.py` - Original MNIST training script
- `Models.py` - Original 2D CNN model
- `train_utils.py` - Training utilities

## Setup

### 1. Download the Dataset

You need to download the CSPB.ML.2022R2 dataset:
- **Batch 1**: Contains signal_1.tim through signal_4000.tim
- **Label file**: `signal_record_first_20000.txt` with modulation labels

Download from: [CSP Blog - CSPB.ML.2022R2](https://cyclostationary.blog/2023/10/02/cspb-ml-2022r2-correcting-an-rng-flaw-in-cspb-ml-2022/)

### 2. Install Dependencies

```bash
pip install torch torchvision numpy matplotlib
```

### 3. Organize Data

Place your downloaded files in the project directory:
```
prototype_learning_signals_dev/
├── Batch_Dir_1/           # Your downloaded .tim files
│   ├── signal_1.tim
│   ├── signal_2.tim
│   └── ...
├── signal_record_first_20000.txt  # Label file
└── [other project files]
```

## Usage

### 1. Training

Run the training script:

```bash
python rf_main.py --data_dir Batch_Dir_1 --label_file signal_record_first_20000.txt
```

**Optional parameters:**
- `--lr 0.01` - Learning rate
- `--batch_size 32` - Batch size
- `--num_classes 8` - Number of modulation classes
- `--h 2` - Hidden layer dimension (feature space)
- `--scale 2` - Distance scaling factor
- `--reg 0.001` - Regularization coefficient

### 2. Visualization

After training, visualize the learned prototypes:

```bash
python rf_visualize.py
```

This will create:
- `rf_prototypes.png` - Scatter plot showing features clustered around prototypes
- `rf_signal_examples.png` - Example signals from each modulation type

## How It Works

### 1. Data Processing
- **Binary reading**: `.tim` files contain interleaved real/imaginary parts
- **Feature extraction**: Convert complex signals to magnitude and phase channels
- **Normalization**: Scale features to [0,1] range

### 2. Model Architecture
- **1D CNN**: Three convolutional blocks with max pooling
- **Feature space**: Maps to 2D space for visualization
- **DCE loss**: Learns prototype centers for each modulation type

### 3. Prototype Learning
- **Centers**: Learnable parameters representing ideal features for each class
- **Distance-based classification**: Classify based on proximity to prototypes
- **Regularization**: Pull features closer to their class prototypes

### 4. Training Process
1. Extract features using 1D CNN
2. Compute distances to prototype centers
3. Apply DCE loss for classification
4. Apply regularization to improve clustering
5. Update both CNN weights and prototype positions

## Expected Results

After training, you should see:
- **High accuracy**: >90% classification accuracy
- **Clear clustering**: Features from same modulation type cluster together
- **Well-separated prototypes**: Each modulation type has distinct prototype location

## Troubleshooting

### Missing Label File
If you don't have the label file, the script will create a dummy one for testing:
```bash
python rf_main.py --data_dir Batch_Dir_1
```

### Memory Issues
Reduce batch size if you encounter memory problems:
```bash
python rf_main.py --batch_size 16
```

### GPU Usage
The code automatically uses GPU if available. For CPU-only:
```python
device = 'cpu'  # In rf_visualize.py
```

## Key Differences from MNIST

1. **1D vs 2D**: RF signals use 1D convolutions instead of 2D
2. **Complex signals**: Convert complex data to magnitude/phase representation
3. **8 classes**: RF has 8 modulation types vs MNIST's 10 digits
4. **Signal length**: Variable-length signals vs fixed 28x28 images

## References

- Original paper: [Robust Classification with Convolutional Prototype Learning](https://arxiv.org/abs/1805.03438)
- Dataset: [CSPB.ML.2022R2](https://cyclostationary.blog/2023/10/02/cspb-ml-2022r2-correcting-an-rng-flaw-in-cspb-ml-2022/)
- CSP Blog: [Cyclostationary Signal Processing](https://cyclostationary.blog/) 