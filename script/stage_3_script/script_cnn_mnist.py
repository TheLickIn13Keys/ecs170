from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader

if __name__ == '__main__':
    data_obj = Dataset_Loader('CIFAR', '')


    data_obj.dataset_source_folder_path = str(
        PROJECT_ROOT / 'data/stage_3_data'
    ) + '/'

    data_obj.dataset_source_file_name = 'CIFAR'

    data = data_obj.load()

    print('Training samples:', len(data['train']['X']))
    print('Testing samples:', len(data['test']['X']))
    print('Image shape:', data['train']['X'][0].shape)
    print('First label:', data['train']['y'][0])