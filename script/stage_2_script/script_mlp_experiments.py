from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_2_code.Plot_Learning_Curve import plot_learning_curves

import numpy as np
import torch

DEFAULT = dict(hidden_dims=(256, 128), loss='cross_entropy',
               optimizer='adam', learning_rate=1e-3, max_epoch=20, batch_size=128)

EXPERIMENTS = {
    'depth': {
        '1 hidden (128), full batch': dict(hidden_dims=(128,), batch_size=None),
        '1 hidden (128)': dict(hidden_dims=(128,)),
        '2 hidden (256-128)': dict(),
        '3 hidden (512-256-128)': dict(hidden_dims=(512, 256, 128)),
    },
    'loss': {
        'cross entropy': dict(),
        'multi margin (hinge)': dict(loss='multi_margin'),
        'MSE on softmax': dict(loss='mse'),
    },
    'optimizer': {
        'Adam lr=1e-3': dict(),
        'SGD momentum lr=1e-2': dict(optimizer='sgd', learning_rate=1e-2),
        'RMSprop lr=1e-3': dict(optimizer='rmsprop'),
        'Adagrad lr=1e-2': dict(optimizer='adagrad', learning_rate=1e-2),
    },
}


if __name__ == "__main__":
    result_dir = PROJECT_ROOT / 'result/stage_2_result/experiments'
    result_dir.mkdir(parents=True, exist_ok=True)

    data_obj = Dataset_Loader('stage 2 dataset', '')
    data_obj.dataset_source_folder_path = str(PROJECT_ROOT / 'data/stage_2_data') + '/'
    setting_obj = Setting_Train_Test_Split('train test', '')
    evaluate_obj = Evaluate_Metrics('accuracy, precision, recall, f1', '')

    finished = {}
    rows = []
    for group, runs in EXPERIMENTS.items():
        histories = {}
        for label, overrides in runs.items():
            config = {**DEFAULT, **overrides}
            key = repr(sorted(config.items()))
            if key not in finished:
                print('************ {} / {} ************'.format(group, label))
                np.random.seed(2)
                torch.manual_seed(2)
                method_obj = Method_MLP('multi-layer perceptron', label, **config)
                result_obj = Result_Saver('saver', '')
                result_obj.result_destination_folder_path = str(result_dir) + '/'
                result_obj.result_destination_file_name = 'MLP_{}_{}'.format(group, len(histories) + 1)
                setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
                scores = setting_obj.load_run_save_evaluate()
                finished[key] = (result_obj.data, scores)
            history, scores = finished[key]
            histories[label] = history
            rows.append((group, label, scores))
        plot_learning_curves(histories, str(result_dir / 'learning_curve_{}.png'.format(group)),
                             title='MLP learning convergence: {}'.format(group))

    metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted',
               'precision_macro', 'recall_macro', 'f1_macro', 'f1_micro']
    lines = ['| group | configuration | ' + ' | '.join(metrics) + ' |',
             '|' + '---|' * (len(metrics) + 2)]
    for group, label, scores in rows:
        lines.append('| {} | {} | '.format(group, label) +
                     ' | '.join('{:.4f}'.format(scores[m]) for m in metrics) + ' |')
    table = '\n'.join(lines)
    print('************ Summary ************')
    print(table)
