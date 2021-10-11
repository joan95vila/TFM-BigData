# -*- coding: utf-8 -*-

from utilities import debug, constants
from utilities.param_grid_conf import param_grid
from utilities.constants import PATH_SAVED_CLASSIFIERS
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pickle


class Classifiers:

    def __init__(self, embedding, seed=1):
        self.feature_vectors_train = embedding.feature_vectors_train
        self.targets_train = embedding.targets_train
        self.files_train = embedding.files_train
        self.feature_vectors_validation = embedding.feature_vectors_validation
        self.targets_validation = embedding.targets_validation
        self.files_validation = embedding.files_validation

        self.embedding_name = embedding.embedding.embedding_params["embedding_name"]
        self.embedding_conf_name = embedding.embedding.embedding_params["conf_name"]
        self.classifier_name = None

        self.classifier = None
        self.classifier_results = None
        self.__base_clf = None
        self.seed = seed

    def classify_sentence(self, sentence):
        X_train = self.learningSets.embedding.sentence_to_vector(sentence)
        y_pred = [constants.CATEGORIES[pred_index] for pred_index in self.classifier.predict([X_train])]

        print(y_pred)

    def train_multiple_classifiers(self, classifier_names, percentage_params_try=10, cv=5, training_time=None):

        classifier_aux = self.classifier
        classifier_results_aux = self.classifier_results
        base_clf_aux = self.__base_clf

        total_classifiers = len(classifier_names)
        n = 0
        for classifier_name in classifier_names:
            n += 1
            self.set_classifier(classifier_name)

            n_iter = sum([len(nested_list) for nested_list in list(param_grid[classifier_name].values())])
            print(f"{' '*3}({n}/{total_classifiers}, {round(100*n/total_classifiers, 2)}% "
                  f"of classifiers training completed.)")
            self.train_classifier(param_grid[classifier_name], n_iter=n_iter * percentage_params_try, cv=cv,
                                  training_time=training_time)

        self.classifier = classifier_aux
        self.classifier_results = classifier_results_aux
        self.__base_clf = base_clf_aux

        return training_time

    def train_classifier(self, param_grid, display_best_results=False, n_top=5, n_iter=10, cv=5, training_time=None):
        import time
        import multiprocessing

        scoring = ['accuracy',
                   'f1_macro', 'f1_micro', 'f1_weighted',
                   'precision_macro', 'precision_micro', 'precision_weighted',
                   'recall_macro', 'recall_micro', 'recall_weighted']

        # random search constructor
        n_jobs = int(multiprocessing.cpu_count()*0.75)
        classifier_models = RandomizedSearchCV(estimator=self.__base_clf, param_distributions=param_grid,
                                               scoring=scoring, n_iter=n_iter, cv=cv, random_state=self.seed,
                                               n_jobs=n_jobs, refit='accuracy', verbose=10)

        # classifier train
        print(f" > initializing ({type(self.__base_clf).__name__}) CLASSIFIER randomized search "
              f"TRAINING with the configuration ({self.embedding_conf_name}) of the embedding ({self.embedding_name}).")
        print(f"{' ' * 3}>> Training started.")

        start = time.time()
        classifier_models.fit(X=self.feature_vectors_train, y=self.targets_train)
        end = time.time()

        print(f"{' ' * 3}>> Training completed.")

        # recording the training time information
        if training_time:
            training_time_conf = training_time[self.classifier_name][self.embedding_name]
            if self.embedding_conf_name in training_time_conf["conf_name"]:
                index = training_time_conf["conf_name"].index(self.embedding_conf_name)
                training_time_conf["duration"][index] = round(time.time() - start, 2)
            else:
                training_time_conf["conf_name"].append(self.embedding_conf_name)
                training_time_conf["duration"].append(round(end - start, 2))

        # display the best results
        if display_best_results:
            self.report_best_classifier_models(n_top=n_top)

        # selecting the best classifier and evaluate its performance
        best_classifier = classifier_models.best_estimator_

        # save the classifier
        path = f"{PATH_SAVED_CLASSIFIERS}{self.embedding_name}_{self.embedding_conf_name}" \
               f"_{type(self.__base_clf).__name__}.pickle"
        pickle.dump(best_classifier, open(path, 'wb'))

        path = f"{PATH_SAVED_CLASSIFIERS}{self.embedding_name}_{self.embedding_conf_name}" \
               f"_{type(self.__base_clf).__name__}_classifier_models_results.pickle"
        pickle.dump(classifier_models.cv_results_, open(path, 'wb'))

        # set classifier
        self.classifier = best_classifier
        self.classifier_results = classifier_models.cv_results_

    def load_classifier(self, verbose=False):
        if verbose:
            print(f" > LOADING ({type(self.__base_clf).__name__}) CLASSIFIER randomized search with the configuration "
                  f"({self.embedding_conf_name}) of the \n{' ' * 3}embedding ({self.embedding_name}).")
            print(f"{' ' * 3}>> Started loading.")

        # embedding_conf_name
        path = f"{PATH_SAVED_CLASSIFIERS}{self.embedding_name}_{self.embedding_conf_name}" \
               f"_{type(self.__base_clf).__name__}.pickle"
        self.classifier = pickle.load(open(path, 'rb'))

        path = f"{PATH_SAVED_CLASSIFIERS}{self.embedding_name}_{self.embedding_conf_name}" \
               f"_{type(self.__base_clf).__name__}_classifier_models_results.pickle"
        self.classifier_results = pickle.load(open(path, 'rb'))

        if verbose:
            print(f"{' ' * 3}>> Completed loading.")
            print()

    def set_classifier(self, classifier_name):
        self.classifier_name = classifier_name

        if classifier_name == "linear regression":
            from sklearn.linear_model import Lasso
            self.__base_clf = Lasso()

        elif classifier_name == "logistic regression":
            from sklearn.linear_model import LogisticRegression
            self.__base_clf = LogisticRegression(n_jobs=-1)

        elif classifier_name == "k-neighborhood":
            from sklearn.neighbors import KNeighborsClassifier
            self.__base_clf = KNeighborsClassifier()

        elif classifier_name == "decision tree classifier":
            from sklearn.tree import DecisionTreeClassifier
            self.__base_clf = DecisionTreeClassifier()

        elif classifier_name == "random forest":
            from sklearn.ensemble import RandomForestClassifier
            self.__base_clf = RandomForestClassifier()

        elif classifier_name == "support vector":
            from sklearn.svm import SVC
            self.__base_clf = SVC()

        # print("Model " + classifier_name + " settled.")

    def confusion_matrix(self):
        from sklearn.metrics import plot_confusion_matrix
        from matplotlib import pyplot as plt

        # Plot non-normalized confusion matrix
        body = ""
        titles_options = [("Confusion matrix, without normalization", None),
                          ("Normalized confusion matrix", 'true')]

        # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(50, 10))
        # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))
        fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, frameon=False)
        for [title, normalize], ax in zip(titles_options, axes.flatten()):
            conf_mat = plot_confusion_matrix(self.classifier,
                                             self.feature_vectors_validation,
                                             self.targets_validation,
                                             ax=ax,
                                             cmap='Blues',
                                             display_labels=constants.CATEGORIES.values(),
                                             normalize=normalize,
                                             colorbar=False)

            ax.set_xlabel('Prediction label', fontweight='bold')
            ax.set_ylabel('Real label', fontweight='bold')
            ax.xaxis.label.set_color('purple')
            ax.yaxis.label.set_color('purple')
            # ax.xaxis.set_visible(False)
            # ax.yaxis.set_visible(False)
            ax.title.set_text(title)

            body += f"{'_' * 100}\n{title}\n{'_' * 100}\n{conf_mat.confusion_matrix}\n\n"

        plt.tight_layout()
        plt.show()

        body = body[:-2]
        title = "CONFUSION MATRIX INFORMATION"
        debug.information_block(title, body)

    def report_best_classifier_model_summary(self, classifier_names):
        from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

        base_clf_aux = self.__base_clf

        # sorting
        sort_values = []
        for classifier_name in classifier_names:
            self.set_classifier(classifier_name)
            self.load_classifier()

            y = self.targets_validation
            X = self.feature_vectors_validation
            y_pred = self.classifier.predict(X)

            sort_values.append(round(accuracy_score(y, y_pred), 2))

        classifier_names = [classifier_name for _, classifier_name in reversed(sorted(zip(sort_values,
                                                                                          classifier_names)))]

        body = "\n"
        len_max_name = (max([len(classifier_name) for classifier_name in classifier_names]) + 4)
        border_symbol = '='
        pad = border_symbol * len_max_name
        h_borders = border_symbol
        body += f"  {' ' * len(pad)}{h_borders * (100 - 2)}\n"
        body += f"  {' ' * len(pad)}||{' ' * 7}accuracy{' ' * 7}||{' ' * 10}f1{' ' * 10}||{' ' * 8}recall{' ' * 8}||" \
                f"{' ' * 6}precision{' ' * 7}||\n"
        body += f"  {' ' * len(pad)}||{'-' * (14 + 8)}||{'-' * (20 + 2)}||{'-' * (16 + 6)}||{'-' * (12 + 1 + 9)}||\n"
        body += f"  {' ' * len(pad)}" + f"||{' ' * 2}val.{' ' * 2}|{' ' * 4}test{' ' * 5}" * 4 + "||\n"
        body += f"{pad}{border_symbol * 2}||{border_symbol * (14 + 8)}||{border_symbol * (20 + 2)}||" \
                f"{border_symbol * (16 + 6)}||{border_symbol * (12 + 1 + 9)}||"
        # body += f"\n||{h_borders * (len(pad) + 100-4)}||"
        for classifier_name in classifier_names:
            self.set_classifier(classifier_name)
            self.load_classifier()

            best_result = np.flatnonzero(self.classifier_results['rank_test_accuracy'] == 1)

            y = self.targets_validation
            X = self.feature_vectors_validation
            y_pred = self.classifier.predict(X)

            average = 'macro'

            # validation metrics
            accuracy_validation = round(accuracy_score(y, y_pred), 2)
            f1_validation = round(f1_score(y, y_pred, average=average), 2)
            recall_validation = round(recall_score(y, y_pred, average=average), 2)
            precision_validation = round(precision_score(y, y_pred, average=average), 2)

            # test metrics
            accuracy_test = round(self.classifier_results['mean_test_accuracy'][best_result[0]], 2)
            f1_test = round(self.classifier_results['mean_test_f1_macro'][best_result[0]], 2)
            recall_test = round(self.classifier_results['mean_test_recall_macro'][best_result[0]], 2)
            precision_test = round(self.classifier_results['mean_test_precision_macro'][best_result[0]], 2)
            # test metrics (std)
            accuracy_test_std = round(self.classifier_results['std_test_accuracy'][best_result[0]], 2)
            f1_test_std = round(self.classifier_results['std_test_f1_macro'][best_result[0]], 2)
            recall_test_std = round(self.classifier_results['std_test_recall_macro'][best_result[0]], 2)
            precision_test_std = round(self.classifier_results['std_test_precision_macro'][best_result[0]], 2)

            diff_pad = ' ' * (len_max_name - len(classifier_name) - 2)
            body += f"\n|| {classifier_name} {diff_pad}" \
                    f"||  {accuracy_validation:4}  | {accuracy_test:4} ({accuracy_test_std:4}) " \
                    f"||  {f1_validation:4}  | {f1_test:4} ({f1_test_std:4}) " \
                    f"||  {recall_validation:4}  | {recall_test:4} ({recall_test_std:4}) " \
                    f"||  {precision_validation:4}  | {precision_test:4} ({precision_test_std:4}) " \
                    f"||" \
                    f"\n||{pad}" + f"||{'-' * 8}|{'-' * 13}" * 4 + "||" \
                # f"\n||{'-' * (len(pad) + 100 - 4)}||"
        body = body[:-(len(pad) + 100 + 1)]
        body += f"\n{h_borders * (len(pad) + 100)}\n"

        title = f"MODEL METRICS (embedding: {self.embedding_name}, configuration: {self.embedding_conf_name})"
        debug.information_block(title, body)

        self.set_classifier(type(base_clf_aux).__name__)
        self.load_classifier()

    # https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules
    def report_best_classifier_models(self, n_top=3):
        ALTERNATIVE_FORMAT = False

        results = self.classifier_results
        str_body = ""
        for i in range(n_top, 0, -1):
            candidates = np.flatnonzero(results['rank_test_accuracy'] == i)
            for candidate in candidates:
                # rank
                str_body += f"{'-' * 100}\nModel with rank -> {i}\n{'-' * 100}\n"

                if ALTERNATIVE_FORMAT:
                    # accuracy
                    print(f"accuracy\t\t -> mean: {round(results['mean_test_accuracy'][candidate], 2)} | "
                          f"std: {round(results['std_test_accuracy'][candidate], 2)}")

                    # f1 macro
                    print(f"f1 macro\t\t -> mean: {round(results['mean_test_f1_macro'][candidate], 2)} | "
                          f"std: {round(results['std_test_f1_macro'][candidate], 2)}")
                    # f1 micro
                    print(f"f1 macro\t\t -> mean: {round(results['mean_test_f1_macro'][candidate], 2)} | "
                          f"std: {round(results['std_test_f1_micro'][candidate], 2)}")

                    # precision macro
                    print(f"precision macro  -> mean: {round(results['mean_test_precision_macro'][candidate], 2)} | "
                          f"std: {round(results['std_test_precision_macro'][candidate], 2)}")
                    # precision micro
                    print(f"precision micro  -> mean: {round(results['mean_test_precision_micro'][candidate], 2)} | "
                          f"std: {round(results['std_test_precision_micro'][candidate], 2)}")

                    # recall macro
                    print(f"recall macro\t -> mean: {round(results['mean_test_recall_macro'][candidate], 2)} | "
                          f"std: {round(results['std_test_recall_macro'][candidate], 2)}")
                    # recall micro
                    print(f"recall micro\t -> mean: {round(results['mean_test_recall_micro'][candidate], 2)} | "
                          f"std: {round(results['std_test_recall_micro'][candidate], 2)}")

                    # parameters
                    print(f"Parameters\t\t -> {results['params'][candidate]}\n")

                else:
                    # accuracy
                    str_body += f"mean: {round(results['mean_test_accuracy'][candidate], 2)} | " \
                                f"std: {round(results['std_test_accuracy'][candidate], 2)} " \
                                f"(accuracy)\n"

                    # # f1 weighted
                    # str_body += f"mean: {round(results['mean_test_f1_weighted'][candidate], 2)} | " \
                    #             f"std: {round(results['std_test_f1_weighted'][candidate], 2)} " \
                    #             f"(f1 weighted)\n"
                    # f1 macro
                    str_body += f"mean: {round(results['mean_test_f1_macro'][candidate], 2)} | " \
                                f"std: {round(results['std_test_f1_macro'][candidate], 2)} " \
                                f"(f1 macro)\n"
                    # f1 micro
                    str_body += f"mean: {round(results['mean_test_f1_micro'][candidate], 2)} | " \
                                f"std: {round(results['std_test_f1_micro'][candidate], 2)} " \
                                f"(f1 micro)\n"

                    # # precision weighted
                    # str_body += f"mean: {round(results['mean_test_precision_weighted'][candidate], 2)} | " \
                    #             f"std: {round(results['std_test_precision_weighted'][candidate], 2)} " \
                    #             f"(precision weighted)\n"
                    # precision macro
                    str_body += f"mean: {round(results['mean_test_precision_macro'][candidate], 2)} | " \
                                f"std: {round(results['std_test_precision_macro'][candidate], 2)} " \
                                f"(precision macro)\n"
                    # precision micro
                    str_body += f"mean: {round(results['mean_test_precision_micro'][candidate], 2)} | " \
                                f"std: {round(results['std_test_precision_micro'][candidate], 2)} " \
                                f"(precision micro)\n"

                    # # recall weighted
                    # str_body += f"mean: {round(results['mean_test_recall_weighted'][candidate], 2)} | " \
                    #             f"std: {round(results['std_test_recall_weighted'][candidate], 2)} " \
                    #             f"(recall weighted)\n"
                    # recall macro
                    str_body += f"mean: {round(results['mean_test_recall_macro'][candidate], 2)} | " \
                                f"std: {round(results['std_test_recall_macro'][candidate], 2)} " \
                                f"(recall macro)\n"
                    # recall micro
                    str_body += f"mean: {round(results['mean_test_recall_micro'][candidate], 2)} | " \
                                f"std: {round(results['std_test_recall_micro'][candidate], 2)} " \
                                f"(recall macro)\n"

                    # parameters
                    str_body += f"\nParameters -> {results['params'][candidate]}\n\n"

        str_body = str_body[:-2]
        title = F"TOP {n_top} BEST ESTIMATED MODELS INFORMATION"
        debug.information_block(title, str_body)

    def display_accuracy(self):
        from sklearn.metrics import accuracy_score, f1_score

        y_pred = self.classifier.predict(self.feature_vectors_validation)
        accuracy_percentage = round(accuracy_score(self.targets_validation, y_pred) * 100, 2)
        f1_score_percentage = round(f1_score(self.targets_validation, y_pred, average='weighted'), 4)

        title = f"ACCURACY INFORMATION ({type(self.__base_clf).__name__})"
        body = f"Testing accuracy: {accuracy_percentage} ({round(100 - accuracy_percentage, 2)}% of errors)\n" \
               f"Testing F1 score: {f1_score_percentage}"
        debug.information_block(title, body)

    def display_errors(self):
        y_pred = self.classifier.predict(self.feature_vectors_validation)

        categories = ['business', 'entertainment', 'politics', 'sport', 'tech']
        categories_map = {i: category for i, category in enumerate(categories)}

        n = 0
        str_body = ""
        for i, [y, pred] in enumerate(zip(self.targets_validation, y_pred)):
            if y != pred:
                str_body += f"Number: {i} ({categories_map[y]}-{self.files_validation[i]})\n" \
                            f"y:\t  {categories_map[y]}\n" \
                            f"pred: {categories_map[pred]}\n\n"
                n += 1

        title = f"ERRORS ({type(self.__base_clf).__name__})"
        body = f"{str_body}" \
               f"{'-' * 100}\nNumber of errors: {n} ({round(n / len(y_pred) * 100, 2)}% of total data to predict)"
        debug.information_block(title, body)
