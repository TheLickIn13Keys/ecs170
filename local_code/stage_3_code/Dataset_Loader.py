import pickle
import numpy as np
from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):

    def load(self):
        print('loading data...')

        file_path = (
            self.dataset_source_folder_path
            + self.dataset_source_file_name
        )

        # Load binary pickle file
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        result = {}

        for split in ('train', 'test'):
            X = []
            y = []

            for instance in data[split]:
                X.append(np.array(instance['image']))
                y.append(int(instance['label']))

            result[split] = {
                'X': np.array(X),
                'y': np.array(y)
            }

        return result