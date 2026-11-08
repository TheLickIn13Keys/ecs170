
from pathlib import Path
import sys
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



if __name__ == '__main__':
    torch.manual_seed(2)

    # Load CIFAR-10
    data_obj = Dataset_Loader('CIFAR', '')
    data_obj.dataset_source_folder_path = str(
        PROJECT_ROOT / 'data/stage_3_data'
    ) + '/'
    data_obj.dataset_source_file_name = 'CIFAR'

    data = data_obj.load()

    # Create CNN
    model = Method_CNN_CIFAR('CNN CIFAR', '')
    model.data = data

    # Train and test
    result = model.run()

    y_true = result['true_y']
    y_pred = result['pred_y']

    print(f"Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")
    print(f"Precision: {precision_score(y_true, y_pred, average='macro', zero_division=0) * 100:.2f}%")
    print(f"Recall: {recall_score(y_true, y_pred, average='macro', zero_division=0) * 100:.2f}%")
    print(f"F1-score: {f1_score(y_true, y_pred, average='macro', zero_division=0) * 100:.2f}%")

    pred_y = result['pred_y']
    true_y = result['true_y']

    accuracy = (pred_y == true_y).mean()
    print(f'CIFAR Test Accuracy: {accuracy:.4f}')

    # Save learning curves
    output_dir = PROJECT_ROOT / 'result/stage_3_result'
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(result['train_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('CIFAR CNN Training Loss')
    plt.savefig(output_dir / 'CIFAR_loss.png')
    plt.close()

    plt.figure()
    plt.plot(result['train_acc'])
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.title('CIFAR CNN Training Accuracy')
    plt.savefig(output_dir / 'CIFAR_accuracy.png')
    plt.close()

    print('Learning curves saved.')
