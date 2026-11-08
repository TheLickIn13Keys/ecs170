
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_MNIST_Experiment import (
    Method_CNN_MNIST_Experiment
)

# Load MNIST once
data_obj = Dataset_Loader('MNIST', '')
data_obj.dataset_source_folder_path = str(PROJECT_ROOT / 'data/stage_3_data') + '/'
data_obj.dataset_source_file_name = 'MNIST'

data = data_obj.load()

# MNIST CNN ablation experiments
experiments = [
    # Experiment 1: Kernel size (3x3 -> 5x5)
    {'name': 'Kernel_5x5', 'kernel_size': 5, 'layers': 2},

    # Experiment 2: CNN depth (2 -> 3 layers)
    {'name': 'Depth_3', 'kernel_size': 3, 'layers': 3},
]

results = {}

for config in experiments:
    print(f"\nRunning experiment: {config['name']}")

    torch.manual_seed(2)
    np.random.seed(2)

    model = Method_CNN_MNIST_Experiment(
        config['name'],
        '',
        kernel_size=config['kernel_size'],
        num_conv_layers=config['layers']
    )

    model.data = data
    model.max_epoch = 10

    result = model.run()

    accuracy = np.mean(result['pred_y'] == result['true_y'])

    results[config['name']] = {
        'test_accuracy': accuracy,
        'train_loss': result['train_loss'],
        'train_acc': result['train_acc']
    }

    print(f"{config['name']} Test Accuracy: {accuracy:.4f}")

print("\n=== MNIST Ablation Results ===")

for name, result in results.items():
    print(f"{name}: {result['test_accuracy']:.4f}")
