# -*- coding: utf-8 -*-

from utilities.constants import PATH_EXECUTION_TIMINGS
from utilities.param_grid_conf import param_grid
from utilities.embeddings_params import *
from utilities.constants import *
from models.classifiers import Classifiers
from models.embeddings import *

import random
import time


def train_embeddings_classifiers(train_embeddings=True, train_classifiers=True):
    # To replicate the results
    random.seed(1)

    # TRAIN SETTINGS
    # ==================================================================================================================
    EMBEDDING_TRAINING = train_embeddings
    CLASSIFIERS_TRAIN = train_classifiers

    # ==================================================================================================================

    # EMBEDDINGS AND CLASSIFIERS DURATION INFORMATION (INITIALIZATION)
    # ==================================================================================================================
    def __execution_timings_embeddings_dictionary_builder(embedding_names):
        return \
            {
                embedding: {
                    "conf_name": [],
                    "duration": [],
                } for embedding in embedding_names
            }

    def __execution_timings_classifiers_dictionary_builder(embedding_names, classifier_names):
        return \
            {
                classifier: {
                    embedding: {
                        "conf_name": [],
                        "duration": []
                    } for embedding in embedding_names
                } for classifier in classifier_names
            }

    def __execution_timings_classifiers_embedding_dictionary_updater(embedding_names, classifier_names,
                                                                     timings_classifiers):
        for classifier in classifier_names:
            timings_classifiers[classifier].update({
                embedding: {
                    "conf_name": [],
                    "duration": []
                } for embedding in embedding_names
            })

    # loading the timing information
    if EMBEDDING_TRAINING:
        try:
            path = f"{PATH_EXECUTION_TIMINGS}embeddings_timings.pickle"
            execution_timings_embeddings = pickle.load(open(path, 'rb'))

            new_embeddings = list(set(EMBEDDING_NAMES) - set(execution_timings_embeddings.keys()))
            if new_embeddings:
                execution_timings_embeddings.update(__execution_timings_embeddings_dictionary_builder(new_embeddings))
        except FileNotFoundError:
            execution_timings_embeddings = __execution_timings_embeddings_dictionary_builder(EMBEDDING_NAMES)
            pickle.dump(execution_timings_embeddings, open(path, 'wb'))

    if CLASSIFIERS_TRAIN:
        # --------------------------------------------------------------------------------------------------------------
        new_embeddings = None
        new_classifiers = None
        if not EMBEDDING_TRAINING:
            try:  # load embedding timings
                path = f"{PATH_EXECUTION_TIMINGS}embeddings_timings.pickle"
                execution_timings_embeddings = pickle.load(open(path, 'rb'))

            except FileNotFoundError:
                print("Execution finalized, no embedding time file found.")
                exit

        try:  # load classifier timings
            path = f"{PATH_EXECUTION_TIMINGS}classifiers_timings.pickle"
            execution_timings_classifiers = pickle.load(open(path, 'rb'))

            new_classifiers = list(set(CLASSIFIER_NAMES) - set(execution_timings_classifiers.keys()))
            new_embeddings = list(set(EMBEDDING_NAMES) -
                                  set([*execution_timings_classifiers[[*execution_timings_classifiers][0]]]))

            if new_classifiers and new_embeddings:
                execution_timings_classifiers.update(
                    __execution_timings_classifiers_dictionary_builder(new_embeddings, new_classifiers)
                )

            elif new_embeddings:
                __execution_timings_classifiers_embedding_dictionary_updater(new_embeddings, CLASSIFIER_NAMES,
                                                                             execution_timings_classifiers)
                # execution_timings_classifiers.update(
                #     __execution_timings_classifiers_dictionary_builder(new_embeddings, CLASSIFIER_NAMES)
                # )

            elif new_classifiers:
                execution_timings_classifiers.update(
                    __execution_timings_classifiers_dictionary_builder(EMBEDDING_NAMES, new_classifiers)
                )

        except FileNotFoundError:
            execution_timings_classifiers = __execution_timings_classifiers_dictionary_builder(EMBEDDING_NAMES,
                                                                                               CLASSIFIER_NAMES)

        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # try:
        #     path = f"{PATH_EXECUTION_TIMINGS}embeddings_timings.pickle"
        #     execution_timings_embeddings = pickle.load(open(path, 'rb'))
        #
        #     path = f"{PATH_EXECUTION_TIMINGS}classifiers_timings.pickle"
        #     execution_timings_classifiers = pickle.load(open(path, 'rb'))
        #
        #     new_embeddings = list(set(EMBEDDING_NAMES) - set(execution_timings_embeddings.keys()))
        #     new_classifiers = list(set(CLASSIFIER_NAMES) - set(execution_timings_classifiers.keys()))
        #     if new_classifiers and new_embeddings:
        #         execution_timings_classifiers = \
        #             __execution_timings_classifiers_dictionary_builder(new_embeddings, new_classifiers)
        #     elif new_classifiers:
        #         execution_timings_classifiers = \
        #             __execution_timings_classifiers_dictionary_builder(EMBEDDING_NAMES, new_classifiers)
        #     elif new_embeddings:
        #         execution_timings_classifiers = \
        #             __execution_timings_classifiers_dictionary_builder(new_embeddings, CLASSIFIER_NAMES)
        # except FileNotFoundError:
        #     execution_timings_classifiers = __execution_timings_classifiers_dictionary_builder(EMBEDDING_NAMES,
        #                                                                                        CLASSIFIER_NAMES)
        #     pickle.dump(execution_timings_classifiers, open(path, 'wb'))
    # --------------------------------------------------------------------------------------------------------------

    # ==================================================================================================================

    # EMBEDDING SELECTION AND TRAINING
    # ==================================================================================================================
    for embedding_name in EMBEDDING_NAMES:
        for param_combination in EMBEDDING_PARAMS[embedding_name]:

            if EMBEDDING_TRAINING:
                start = time.time()

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

            print(f"\n\n{'-' * 100}")
            print(f"EMBEDDING ({embedding_name}) & CONFIGURATION ({param_combination['conf_name']})\n{'-' * 100}")

            try:
                embedding = LearningSets(embedding, train=EMBEDDING_TRAINING)
                print()

                if EMBEDDING_TRAINING:
                    print(execution_timings_embeddings)
                    training_time_conf = execution_timings_embeddings[embedding_name]
                    if param_combination["conf_name"] in execution_timings_embeddings[embedding_name]["conf_name"]:
                        index = training_time_conf["conf_name"].index(param_combination["conf_name"])
                        training_time_conf["duration"][index] = round(time.time() - start, 2)
                    else:
                        training_time_conf["conf_name"].append(param_combination["conf_name"])
                        training_time_conf["duration"].append(round(time.time() - start, 2))
                # ======================================================================================================

                # CLASSIFIERS SELECTION AND TRAINING
                # ======================================================================================================
                classifiers = Classifiers(embedding)
                classifiers_names = CLASSIFIER_NAMES

                if CLASSIFIERS_TRAIN:
                    execution_timings_classifiers = classifiers.train_multiple_classifiers(
                        classifiers_names, percentage_params_try=0.25, training_time=execution_timings_classifiers)
                    # classifiers.train_classifier(param_grid[CLASSIFIER_NAMES[2]], n_iter=100)

                # classifiers.report_best_classifier_models(5)
                # classifiers.display_accuracy()
                # classifiers.display_errors()
                # classifiers.confusion_matrix()

                try:
                    classifiers.report_best_classifier_model_summary(classifiers_names)
                except FileNotFoundError:
                    print(f"{' ' * 3}>> ALERT! --> No classifier found.")
                except ValueError as e:
                    print(f"{' ' * 3}>> ALERT ON CLASSIFIER! --> {e}")
                # ======================================================================================================

            except FileNotFoundError:
                print(f"{' ' * 3}>> ALERT! --> No embedding or embedding configuration found.")
            # except ValueError as e:
            #     print(f"{' ' * 3}>> ALERT ON EMBEDDING! --> {e}")

    # EMBEDDINGS AND CLASSIFIERS DURATION INFORMATION (FINALIZATION)
    # ==================================================================================================================
    # saving the timing information
    if EMBEDDING_TRAINING:
        path = f"{PATH_EXECUTION_TIMINGS}embeddings_timings.pickle"
        pickle.dump(execution_timings_embeddings, open(path, 'wb'))

        print(execution_timings_embeddings)

    if CLASSIFIERS_TRAIN:
        path = f"{PATH_EXECUTION_TIMINGS}classifiers_timings.pickle"
        pickle.dump(execution_timings_classifiers, open(path, 'wb'))

        print(execution_timings_classifiers)
    # ==================================================================================================================
