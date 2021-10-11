# -*- coding: utf-8 -*-

import random
from os import listdir


class DatasetOperations:

    def __init__(self, path="../../../DATASETS/BBC News Summary/News Articles Processed") -> None:
        self.categories = ['business', 'entertainment', 'politics', 'sport', 'tech']
        self.categories_map = {category: i for i, category in enumerate(self.categories)}
        self.__create(root_path=path)

    def __create(self, root_path):
        self.dataset = list()

        # dataset = { key : [] for key in categories } # dictionary comprehension

        for category in self.categories:
            path = f"{root_path}\\{category}"
            for filename in listdir(path):
                with open(f"{path}\\{filename}") as file:
                    self.dataset.append((file.read(), category, filename))

        # for i, entry in enumerate(dataset[:4]): print(f"{entry[1].upper()} ({i:03})\n{'-'*100}{entry[0]}\n")
        # print(*dataset[:4], sep="\n"*2)

    def shuffle_sample(self):
        return random.sample(self.dataset, len(self.dataset))                                                           # LOAD_SET establish init dataset size

    def create_training_datasets(self):
        dataset_randomized = self.shuffle_sample()
        index_threshold_80 = int(round(len(dataset_randomized) * 0.8, 0))

        # Train datasets
        X_train, Y_train, file_train = [[] for _ in range(3)]
        training_data = dataset_randomized[:index_threshold_80]
        for entry in training_data: X_train.append(entry[0]), Y_train.append(entry[1]), file_train.append(entry[2])

        # Test datasets
        X_test, Y_test, file_test = [[] for _ in range(3)]
        testing_data = dataset_randomized[index_threshold_80:]
        for entry in testing_data: X_test.append(entry[0]), Y_test.append(entry[1]), file_test.append(entry[2])

        # Labeling (Y datasets) categorization

        Y_train = [self.categories_map[category] for category in Y_train]
        Y_test = [self.categories_map[category] for category in Y_test]

        datasets = [X_train, Y_train, file_train, X_test, Y_test, file_test]
        return datasets
