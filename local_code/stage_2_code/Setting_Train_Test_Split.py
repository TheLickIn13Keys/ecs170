'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting

class Setting_Train_Test_Split(setting):
    train_file_name = 'train.csv'
    test_file_name = 'test.csv'
    loaded_data = None

    def load_run_save_evaluate(self):

        # The files are already partitioned; do not split or cross-validate.
        if self.loaded_data is None:
            # Load train
            self.dataset.dataset_source_file_name = self.train_file_name
            train_data = self.dataset.load()
            #Load test
            self.dataset.dataset_source_file_name = self.test_file_name
            test_data = self.dataset.load()
            #checking we loaded properly
            print('Training samples:', len(train_data['X']), ', Testing samples:', len(test_data['X']),
                  ', Features per sample:', len(train_data['X'][0]))
            self.loaded_data = {'train': train_data, 'test': test_data}

        self.method.data = self.loaded_data
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result
        return self.evaluate.evaluate()
