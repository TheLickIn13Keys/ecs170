from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_CIFAR_Experiment import (
    Method_CNN_CIFAR_Experiment
)

# Load CIFAR data once
data_obj = Dataset_Loader('CIFAR', '')
data_obj.dataset_source_folder_path = str(PROJECT_ROOT / 'data/stage_3_data') + '/'
data_obj.dataset_source_file_name = 'CIFAR'

data = data_obj.load()

# Compare kernel sizes
experiments = [
    # Experiment 1: Kernel size (3x3 -> 5x5)
    {'name': 'Kernel_5x5', 'kernel_size': 5, 'pooling': 'max', 'dim': 256, 'layers': 3},

    # Experiment 2: Pooling (MaxPool -> AvgPool)
    {'name': 'Pooling_Avg', 'kernel_size': 3, 'pooling': 'avg', 'dim': 256, 'layers': 3},

    # Experiment 3: Hidden dimension (256 -> 128)
    {'name': 'Hidden_128', 'kernel_size': 3, 'pooling': 'max', 'dim': 128, 'layers': 3},

    # Experiment 4: CNN depth (3 -> 2 convolutional layers)
    {'name': 'Depth_2', 'kernel_size': 3, 'pooling': 'max', 'dim': 256, 'layers': 2},

    # Experiment 5: Loss function (CrossEntropy -> MultiMargin)
    {'name': 'Loss_MultiMargin', 'kernel_size': 3, 'pooling': 'max', 'dim': 256, 'layers': 3, 'loss': 'multi_margin'},
]


results = {}

for config in experiments:
    print(f"\nRunning experiment: {config['name']}")

    torch.manual_seed(2)
    np.random.seed(2)

    model = Method_CNN_CIFAR_Experiment(
        config['name'],
        '',
        kernel_size=config['kernel_size'],
        pooling=config['pooling'],
        hidden_dim=config['dim'],
        num_conv_layers=config['layers'],
        loss=config.get('loss', 'cross_entropy'),
    )

    model.data = data
    model.max_epoch = 20

    result = model.run()

    accuracy = np.mean(result['pred_y'] == result['true_y'])

    results[config['name']] = {
        'test_accuracy': accuracy,
        'train_loss': result['train_loss'],
        'train_acc': result['train_acc']
    }

    print(f"{config['name']} Test Accuracy: {accuracy:.4f}")

# Print comparison
print("\n=== CIFAR Ablation Results ===")

for name, result in results.items():
    print(f"{name}: {result['test_accuracy']:.4f}")
