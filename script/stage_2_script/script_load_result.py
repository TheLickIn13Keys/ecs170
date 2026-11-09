from pathlib import Path
import sys
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from local_code.stage_2_code.Result_Loader import Result_Loader
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics


def save_metrics_table(rows, file_path):
    """Save the report's comparison metrics without rerunning training."""
    metrics = list(rows[0][2])
    with open(file_path, 'w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['group', 'configuration'] + metrics)
        for group, label, scores in rows:
            writer.writerow([group, label] + [scores[name] for name in metrics])


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

    # Reuse the experiment definitions to associate each saved result with its settings.
    from script.stage_2_script.script_mlp_experiments import DEFAULT, EXPERIMENTS
    result_dir = PROJECT_ROOT / 'result/stage_2_result/experiments'
    finished = {}
    rows = []
    for group, runs in EXPERIMENTS.items():
        for number, (label, overrides) in enumerate(runs.items(), start=1):
            key = repr(sorted({**DEFAULT, **overrides}.items()))
            if key not in finished:
                result_obj.result_destination_folder_path = str(result_dir) + '/'
                result_obj.result_destination_file_name = 'MLP_{}_{}'.format(group, number)
                result_obj.load()
                evaluate_obj.data = result_obj.data
                finished[key] = evaluate_obj.evaluate()
            rows.append((group, label, finished[key]))
    save_metrics_table(rows, result_dir / 'metrics.csv')
    print('Saved experiment metrics to', result_dir / 'metrics.csv')
