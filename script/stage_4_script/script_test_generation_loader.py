from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from local_code.stage_4_code.Dataset_Loader_Generation import (
    Dataset_Loader_Generation
)

loader = Dataset_Loader_Generation("Jokes", "")

loader.dataset_source_folder_path = str(
    PROJECT_ROOT / "data/stage_4_data/text_generation/data"
)

data = loader.load()

print("\nFirst joke:")
print(data["jokes"][0])

print("\nFirst input sequence:")
print(data["train"]["X"][0])

print("\nFirst target:")
print(data["id_to_word"][int(data["train"]["y"][0])])
