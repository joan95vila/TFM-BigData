# -*- coding: utf-8 -*-

from utilities import debug
from main_modules.train_module import *
from main_modules.display_information_module import *
from main_modules.display_information_dataset import *
from utilities.constants import CLASSIFIER_NAMES

import random
import time

# To replicate the results
random.seed(1)

# PROGRAM DURATION INFORMATION
# ======================================================================================================================
start = time.time()
print(f"\n{'#' * 100}\nTHE PROGRAM HAS STARTED\n{'#' * 100}")
# ======================================================================================================================

# train_embeddings_classifiers(train_embeddings=False, train_classifiers=False)

REPRESENT_DOC2VEC_Conf3_PCA3_VECTORS_GRAPHIC = False
if REPRESENT_DOC2VEC_Conf3_PCA3_VECTORS_GRAPHIC:
    LearningSets(Doc2Vec(EMBEDDING_PARAMS["Doc2Vec"][2])).visualize_3d()

report_best_models()

# distribution_length_data()

# PROGRAM DURATION INFORMATION
# ======================================================================================================================
total_timing = round(time.time() - start, 2)

title = "PROGRAM DURATION INFORMATION"
body = f"Execution duration: {total_timing} seconds"
debug.information_block(title, body)
# ======================================================================================================================

# from main_modules.display_information import training_timings
# training_timings()
