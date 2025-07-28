# Prototype Learning for RF Signal Classification

This repository contains an implementation of prototype learning for radio frequency (RF) signal classification using PyTorch. The project focuses on classifying different digital modulation schemes from RF signals using convolutional neural networks with prototype learning.

## Features

- **Prototype Learning**: Implementation of Distance-based Cross Entropy (DCE) loss for prototype learning
- **RF Signal Processing**: Support for I/Q, magnitude/phase, and spectrogram features
- **Multiple Datasets**: Support for CSPB.ML.2018R2 and RML2016.10a datasets
- **Improved Models**: Enhanced CNN architectures with batch normalization and dropout
- **Comprehensive Training**: Full training pipeline with monitoring and visualization

## Supported Modulation Schemes

- BPSK (Binary Phase-Shift Keying)
- QPSK (Quadrature Phase-Shift Keying)
- 8PSK (8-Phase Shift Keying)
- π/4-DQPSK (π/4-Differential Quadrature Phase-Shift Keying)
- 16QAM (16-Quadrature Amplitude Modulation)
- 64QAM (64-Quadrature Amplitude Modulation)
- 256QAM (256-Quadrature Amplitude Modulation)
- MSK (Minimum-Shift Keying)

## Project Structure

```
├── rf_main_best.py              # Main training script with best configuration
├── rf_models_improved.py        # Improved CNN models with prototype learning
├── rf_data_loader_improved.py   # Enhanced data loading and preprocessing
├── train_utils.py               # Training utilities and learning rate scheduling
├── rf_visualize.py              # Visualization tools for results
├── RF_README.md                 # Detailed RF-specific documentation
└── upload_to_sagemaker.sh       # Script for AWS SageMaker deployment
```

## Quick Start

### Prerequisites

```bash
pip install torch torchvision numpy scipy matplotlib
```

### Training

```bash
python rf_main_best.py --batch_size 128 --lr 0.0005 --feature_type iq
```

### Key Parameters

- `--batch_size`: Batch size for training (default: 128)
- `--lr`: Learning rate (default: 0.0005)
- `--feature_type`: Feature extraction method: 'iq', 'magnitude_phase', 'spectrogram' (default: 'iq')
- `--max_signals`: Maximum number of signals to load (for testing)
- `--data_dir`: Directory containing signal data

## Model Architecture

The improved model uses:
- **4-layer 1D CNN** with increasing channel dimensions (64→128→256→512)
- **Batch normalization** and **dropout** for regularization
- **Global average pooling** for feature aggregation
- **Prototype learning** with DCE loss for better generalization
- **~1.7M parameters** total

## Results

The model achieves competitive performance on RF signal classification tasks with:
- Improved generalization through prototype learning
- Robust feature extraction from complex RF signals
- Efficient training on CPU (Apple M4) or GPU

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{prototype_learning_rf_2024,
  title={Prototype Learning for RF Signal Classification},
  author={Your Name},
  year={2024},
  url={https://github.com/eddyxwu-ds/proto-learning-rf}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.



