

def distribution_length_data():
    from utilities.constants import PATH_PROCESSED_DATASET, CATEGORIES
    import processing.text.load_dataset as dataset_load

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    dataset_operations = dataset_load.DatasetOperations(path=PATH_PROCESSED_DATASET)
    dataset = dataset_operations.dataset

    dataset_categories = {category: [] for category in [*CATEGORIES.values()]}
    dataset_categories_joined = []

    for data in dataset:
        dataset_categories[data[1]].append(len(data[0]))
        dataset_categories_joined.append(len(data[0]))

    sns.set(rc={'figure.figsize': (20, 15)})
    sns.set_style("whitegrid", rc={'axes.grid': False})
    for i, category in enumerate([*CATEGORIES.values()]):
        sns.displot(dataset_categories[category])
        plt.title(category)


    sns.displot(dataset_categories_joined)
    plt.title("all types of news")

    plt.xlabel("Length of news")
    plt.ylabel("Number of news")
    plt.show()