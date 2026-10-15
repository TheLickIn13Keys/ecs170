from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from local_code.stage_2_code.Result_Loader import Result_Loader
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics


if __name__ == '__main__':
    result_obj = Result_Loader('loader', '')
    result_obj.result_destination_folder_path = str(PROJECT_ROOT / 'result/stage_2_result') + '/'
    result_obj.result_destination_file_name = 'MLP_prediction_result'
    result_obj.fold_count = None
    result_obj.load()

    evaluate_obj = Evaluate_Metrics('accuracy, precision, recall, f1', '')
    evaluate_obj.data = result_obj.data
    print('Test samples:', len(result_obj.data['true_y']))
    for metric, value in evaluate_obj.evaluate().items():
        print('MLP Test {}: {:.4f}'.format(metric, value))
