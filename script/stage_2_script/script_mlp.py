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

#---- Multi-Layer Perceptron script ----
if __name__ == "__main__":
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    #now takes in our stage 2 dataset
    data_obj = Dataset_Loader('stage 2 dataset', '')
    data_obj.dataset_source_folder_path = str(PROJECT_ROOT / 'data/stage_2_data') + '/'

    method_obj = Method_MLP('multi-layer perceptron', '', hidden_dims=(256, 128), activation='relu',
                            loss='cross_entropy', optimizer='adam', learning_rate=1e-3,
                            max_epoch=20, batch_size=128)

    result_dir = PROJECT_ROOT / 'result/stage_2_result'
    result_dir.mkdir(parents=True, exist_ok=True)
    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = str(result_dir) + '/' #results now in stage 2
    result_obj.result_destination_file_name = 'MLP_prediction_result'

    setting_obj = Setting_Train_Test_Split('train test', '') # load the supplied train/test files

    evaluate_obj = Evaluate_Metrics('accuracy, precision, recall, f1', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    scores = setting_obj.load_run_save_evaluate()
    plot_learning_curves({'MLP 784-256-128-10': result_obj.data},
                         str(result_dir / 'MLP_learning_curve.png'))

    print('************ Overall Performance ************')
    for metric, value in scores.items():
        print('MLP Test {}: {:.4f}'.format(metric, value))
    print('************ Finish ************')
    # ------------------------------------------------------
