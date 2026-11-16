from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from local_code.stage_4_code.Dataset_Loader_Classification import (
    Dataset_Loader_Classification
)

loader = Dataset_Loader_Classification("IMDb", "")

loader.dataset_source_folder_path = (
    str(PROJECT_ROOT / "data/stage_4_data/text_classification")
)

data = loader.load()

print("Training X shape:", data["train"]["X"].shape)
print("Training y shape:", data["train"]["y"].shape)
print("Testing X shape:", data["test"]["X"].shape)
print("Testing y shape:", data["test"]["y"].shape)
