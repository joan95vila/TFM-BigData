from models.classifiers import Classifiers
from models.embeddings import *
from utilities.constants import EMBEDDING_NAMES, CLASSIFIER_NAMES
from utilities.embeddings_params import EMBEDDING_PARAMS

import numpy as np


def training_timings_display():
    from utilities.constants import PATH_EXECUTION_TIMINGS
    from utilities import debug
    from pickle import load

    # EMBEDDINGS
    # ==================================================================================================================
    path = f"{PATH_EXECUTION_TIMINGS}embeddings_timings.pickle"
    embeddings_timings = load(open(path, 'rb'))

    title = "EMBEDDINGS INFORMATION"
    body = embeddings_timings
    debug.information_block(title, body)
    # ==================================================================================================================

    # CLASSIFIERS
    # ==================================================================================================================
    path = f"{PATH_EXECUTION_TIMINGS}classifiers_timings.pickle"
    classifiers_timings = load(open(path, 'rb'))

    classifier_names = [*classifiers_timings.keys()]
    embedding_names = [*classifiers_timings[classifier_names[0]]]
    configuration_names = ['conf_1', 'conf_2', 'conf_3', 'conf_4', 'conf_5', 'conf_6']

    for conf in range(len(configuration_names)):
        print(f"> Configuration ({conf})")
        for classifier in classifier_names:
            print(f" >> ({classifier}) classifier")
            for embedding in embedding_names:
                print(f"  >>> ({embedding}) embedding: {classifiers_timings[classifier][embedding]['duration'][conf]}")

    title = "CLASSIFIERS INFORMATION"
    body = classifiers_timings
    debug.information_block(title, body)
    # ==================================================================================================================


def create_file_best_results():
    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

    best_results = {
        embedding_name: {
            classifier_name: {
                "accuracy": {
                    "validation": [],
                    "test": [],
                    "std_test": []
                },
                "f1": {
                    "validation": [],
                    "test": [],
                    "std_test": []
                },
                "recall": {
                    "validation": [],
                    "test": [],
                    "std_test": []
                },
                "precision": {
                    "validation": [],
                    "test": [],
                    "std_test": []
                }
            } for classifier_name in CLASSIFIER_NAMES
        } for embedding_name in EMBEDDING_NAMES
    }

    # Preparation for the function report_best_classifier_model_summary in order to make it work
    # ------------------------------------------------------------------------------------------------------------------
    configuration_names = ['conf_1', 'conf_2', 'conf_3', 'conf_4', 'conf_5', 'conf_6']

    for embedding_name in EMBEDDING_NAMES:
        for classifier_name in CLASSIFIER_NAMES:
            for conf_num, conf_name in enumerate(configuration_names):
                param_combination = EMBEDDING_PARAMS[embedding_name][conf_num]

                if embedding_name == "Doc2Vec":
                    embedding = Doc2Vec(param_combination)
                elif embedding_name == "OneHotVocabularyEncoding":
                    embedding = OneHotVocabularyEncoding(param_combination)
                elif embedding_name == "OneHotCountWordsEncoding":
                    embedding = OneHotCountWordsEncoding(param_combination)
                elif embedding_name == "OneHotBoWEncoding":
                    embedding = OneHotBoWEncoding(param_combination)
                elif embedding_name == "TfidfVectorizer":
                    embedding = TfidfVectorizer(param_combination)

                embedding = LearningSets(embedding, train=False, verbose=False)
                clf = Classifiers(embedding)

                clf.set_classifier(classifier_name)
                clf.load_classifier()
                # ------------------------------------------------------------------------------------------------------------------
                best_result = np.flatnonzero(clf.classifier_results['rank_test_accuracy'] == 1)

                y = clf.targets_validation
                X = clf.feature_vectors_validation
                y_pred = clf.classifier.predict(X)

                average = 'macro'

                # validation metrics
                accuracy_validation = round(accuracy_score(y, y_pred), 2)
                f1_validation = round(f1_score(y, y_pred, average=average), 2)
                recall_validation = round(recall_score(y, y_pred, average=average), 2)
                precision_validation = round(precision_score(y, y_pred, average=average), 2)

                # test metrics
                accuracy_test = round(clf.classifier_results['mean_test_accuracy'][best_result[0]], 2)
                f1_test = round(clf.classifier_results['mean_test_f1_macro'][best_result[0]], 2)
                recall_test = round(clf.classifier_results['mean_test_recall_macro'][best_result[0]], 2)
                precision_test = round(clf.classifier_results['mean_test_precision_macro'][best_result[0]], 2)

                # test metrics (std)
                accuracy_test_std = round(clf.classifier_results['std_test_accuracy'][best_result[0]], 2)
                f1_test_std = round(clf.classifier_results['std_test_f1_macro'][best_result[0]], 2)
                recall_test_std = round(clf.classifier_results['std_test_recall_macro'][best_result[0]], 2)
                precision_test_std = round(clf.classifier_results['std_test_precision_macro'][best_result[0]], 2)

                best_result = best_results[embedding_name][classifier_name]

                best_result['accuracy']['validation'].append(accuracy_validation)
                best_result['accuracy']['test'].append(accuracy_test)
                best_result['accuracy']['std_test'].append(accuracy_test_std)

                best_result['f1']['validation'].append(f1_validation)
                best_result['f1']['test'].append(f1_test)
                best_result['f1']['std_test'].append(f1_test_std)

                best_result['recall']['validation'].append(recall_validation)
                best_result['recall']['test'].append(recall_test)
                best_result['recall']['std_test'].append(recall_test_std)

                best_result['precision']['validation'].append(precision_validation)
                best_result['precision']['test'].append(precision_test)
                best_result['precision']['std_test'].append(precision_test_std)

    path = f"{PATH_SAVED_CLASSIFIERS}__classifiers_best_results.pickle"
    pickle.dump(best_results, open(path, 'wb'))


def load_file_best_results():
    path = f"{PATH_SAVED_CLASSIFIERS}__classifiers_best_results.pickle"
    return pickle.load(open(path, 'rb'))


def report_best_models_without_time():
    from pickle import load

    path = f"{PATH_EXECUTION_TIMINGS}embeddings_timings.pickle"
    embeddings_timings = load(open(path, 'rb'))

    try:
        best_results = load_file_best_results()
    except FileNotFoundError:
        print("The file was not found, so it will be created, it may take some minutes.")
        create_file_best_results()
        best_results = load_file_best_results()

    body = ""

    # Preparation for the function report_best_classifier_model_summary in order to make it work
    # ------------------------------------------------------------------------------------------------------------------
    configuration_names = ['conf_1', 'conf_2', 'conf_3', 'conf_4', 'conf_5', 'conf_6']

    for embedding_name in EMBEDDING_NAMES:
        len_max_name = (max([len(clf_name) for clf_name in CLASSIFIER_NAMES]) + 4)
        border_symbol = '='
        pad = border_symbol * len_max_name
        h_borders = border_symbol

        body += f"  {' ' * len(pad)}{h_borders * (100 - 2)}\n"
        body += f"  {' ' * len(pad)}||{' ' * 7}accuracy{' ' * 7}||{' ' * 10}f1{' ' * 10}||{' ' * 8}recall{' ' * 8}||" \
                f"{' ' * 6}precision{' ' * 7}||\n"
        body += f"embedding:{' ' * 20}||{'-' * (14 + 8)}||{'-' * (20 + 2)}||{'-' * (16 + 6)}||{'-' * (12 + 1 + 9)}||\n"
        body += f"{embedding_name}{' ' * (len_max_name - len(embedding_name) + 2)}" + f"||{' ' * 2}val.{' ' * 2}|" \
                f"{' ' * 4}test{' ' * 5}" * 4 + "||\n"
        body += f"{pad}{border_symbol * 2}||{border_symbol * (14 + 8)}||{border_symbol * (20 + 2)}||" \
                f"{border_symbol * (16 + 6)}||{border_symbol * (12 + 1 + 9)}||"

        for classifier_name in CLASSIFIER_NAMES:
            diff_pad = ' ' * (len_max_name - len(classifier_name) - 2)
            prob_pad_sym = '\\'

            column_pad = f"||{prob_pad_sym * 22}"
            body += f"\n|| {classifier_name} {diff_pad}" \
                    f"{column_pad * 4}" \
                    f"||" \
                    f"\n||{pad}" + f"||{'-' * 8}|{'-' * 13}" * 4 + "||"

            for conf_num, conf_name in enumerate(configuration_names):
                best_result = best_results[embedding_name][classifier_name]

                accuracy_validation = best_result['accuracy']['validation'][conf_num]
                accuracy_test = best_result['accuracy']['test'][conf_num]
                accuracy_test_std = best_result['accuracy']['std_test'][conf_num]

                f1_validation = best_result['f1']['validation'][conf_num]
                f1_test = best_result['f1']['test'][conf_num]
                f1_test_std = best_result['f1']['std_test'][conf_num]

                recall_validation = best_result['recall']['validation'][conf_num]
                recall_test = best_result['recall']['test'][conf_num]
                recall_test_std = best_result['recall']['std_test'][conf_num]

                precision_validation = best_result['precision']['validation'][conf_num]
                precision_test = best_result['precision']['test'][conf_num]
                precision_test_std = best_result['precision']['std_test'][conf_num]

                diff_pad = ' ' * (len_max_name - 6 - 4 - 2)
                body += f"\n|| {' ' * 4}{conf_name} {diff_pad}" \
                        f"||  {accuracy_validation:4}  | {accuracy_test:4} ({accuracy_test_std:4}) " \
                        f"||  {f1_validation:4}  | {f1_test:4} ({f1_test_std:4}) " \
                        f"||  {recall_validation:4}  | {recall_test:4} ({recall_test_std:4}) " \
                        f"||  {precision_validation:4}  | {precision_test:4} ({precision_test_std:4}) " \
                        f"||"
            body += f"\n||{pad}" + f"||{'-' * 8}|{'-' * 13}" * 4 + "||"
        body += body[:-(len(pad) + 100 + 1)]
        body += f"\n{h_borders * (len(pad) + 100)}\n"
        body += "\n"

    body = body[:-(len(pad) + 100 + 2)]
    body += f"{h_borders * (len(pad) + 100)}\n"

    title = f"MODEL METRICS"
    debug.information_block(title, body)

    print(embeddings_timings)

def report_best_models():
    from pickle import load

    path = f"{PATH_EXECUTION_TIMINGS}embeddings_timings.pickle"
    embeddings_timings = load(open(path, 'rb'))

    path = f"{PATH_EXECUTION_TIMINGS}classifiers_timings.pickle"
    classifiers_timings = load(open(path, 'rb'))

    try:
        best_results = load_file_best_results()
    except FileNotFoundError:
        print("The file was not found, so it will be created, it may take some minutes.")
        create_file_best_results()
        best_results = load_file_best_results()

    body = ""

    # Preparation for the function report_best_classifier_model_summary in order to make it work
    # ------------------------------------------------------------------------------------------------------------------
    configuration_names = ['conf_1', 'conf_2', 'conf_3', 'conf_4', 'conf_5', 'conf_6']

    for embedding_name in EMBEDDING_NAMES:
        total_duration = 0

        len_max_name = (max([len(clf_name) for clf_name in CLASSIFIER_NAMES]) + 4)
        border_symbol = '='
        pad = border_symbol * len_max_name
        h_borders = border_symbol

        body += f"{' ' * (len(pad) + 2)}||{h_borders * (22 + 5)}{h_borders * ((22 + 2) * 4)}||\n"
        body += f"  {' ' * len(pad)}||{' ' * 5}training duration{' ' * 5}||{' ' * 7}accuracy{' ' * 7}||" \
                f"{' ' * 10}f1{' ' * 10}||{' ' * 8}recall{' ' * 8}||{' ' * 6}precision{' ' * 7}||\n"
        body += f"embedding:{' ' * 20}" + f"||{'-' * (22 + 5)}" + f"||{'-' * 22}"*4 + "||\n"
        body += f"{embedding_name}{' ' * (len_max_name - len(embedding_name) + 2)}||{' ' * 2}embedding{' ' * 2}|{' ' * 1}classifier{' ' * 2}" \
                + f"||{' ' * 2}val.{' ' * 2}|{' ' * 4}test{' ' * 5}" * 4 + "||\n"
        body += f"{pad}{border_symbol * 2}||{border_symbol * (22 + 5)}" + f"||{border_symbol * 22}"*4 + "||"

        for classifier_name in CLASSIFIER_NAMES:
            diff_pad = ' ' * (len_max_name - len(classifier_name) - 2)
            prob_pad_sym = '\\'

            column_pad = f"||{prob_pad_sym * 22}"
            body += f"\n|| {classifier_name} {diff_pad}||{prob_pad_sym * (22 + 5)}{column_pad * 4}||\n"
            body += f"||{pad}||" + f"{'-' * (22 + 5)}" + f"||{'-' * 8}|{'-' * 13}" * 4 + "||\n"

            for conf_num, conf_name in enumerate(configuration_names):
                best_result = best_results[embedding_name][classifier_name]

                accuracy_validation = best_result['accuracy']['validation'][conf_num]
                accuracy_test = best_result['accuracy']['test'][conf_num]
                accuracy_test_std = best_result['accuracy']['std_test'][conf_num]

                f1_validation = best_result['f1']['validation'][conf_num]
                f1_test = best_result['f1']['test'][conf_num]
                f1_test_std = best_result['f1']['std_test'][conf_num]

                recall_validation = best_result['recall']['validation'][conf_num]
                recall_test = best_result['recall']['test'][conf_num]
                recall_test_std = best_result['recall']['std_test'][conf_num]

                precision_validation = best_result['precision']['validation'][conf_num]
                precision_test = best_result['precision']['test'][conf_num]
                precision_test_std = best_result['precision']['std_test'][conf_num]

                embedding_duration = embeddings_timings[embedding_name]['duration'][conf_num]
                classifier_duration = classifiers_timings[classifier_name][embedding_name]['duration'][conf_num]

                total_duration += classifier_duration

                diff_pad = ' ' * (len_max_name - 6 - 4 - 2)
                body += f"|| {' ' * 4}{conf_name} {diff_pad}" \
                        f"||  {str(embedding_duration).ljust(6)} sec | {str(classifier_duration).ljust(7)} sec " \
                        f"||  {accuracy_validation:4}  | {accuracy_test:4} ({accuracy_test_std:4}) " \
                        f"||  {f1_validation:4}  | {f1_test:4} ({f1_test_std:4}) " \
                        f"||  {recall_validation:4}  | {recall_test:4} ({recall_test_std:4}) " \
                        f"||  {precision_validation:4}  | {precision_test:4} ({precision_test_std:4}) " \
                        f"||\n"
            body += f"||{pad}||" + f"{'-' * (22 + 5)}" + f"||{'-' * 8}|{'-' * 13}" * 4 + "||"
        body = body[:-(len(pad) + 100 + 1 + 29)]
        body += f"\n{h_borders * (len(pad) + 100 + 29)}\n"
        body += f"Total embedding training duration: {round(total_duration, 2)}\n"
        body += '\n'

    title = f"MODEL METRICS"
    debug.information_block(title, body)
