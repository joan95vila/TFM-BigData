# -*- coding: utf-8 -*-


from os import listdir

import time

from utilities import debug

from processing.text.preparation import cleaner


start = time.time()
print(f"\n{'#'*100}\nTHE PROGRAM HAS STARTED\n{'#'*100}\n")

categories = ['business', 'entertainment', 'politics', 'sport', 'tech']

"""
    >> dataset = dict.fromkeys(categories, [])
    No work --> assigns the value [] to all keys, so if we change some value, the values of the others keys will be 
    change as well own to is the same memory slot, is the same object.
"""
# dataset = { key : [] for key in categories } # dictionary comprehension

# root_path = "DATASETS\BBC News Summary\Summaries"
root_path_original_text = "../../../DATASETS/BBC News Summary/News Articles"
root_path_processed_text = "../../../DATASETS/BBC News Summary/News Articles Processed"

print('\\' * 100)
for category in categories:
    print(f" >> The category named: {category} now is beginning to be processed")
    path_original_text = f"{root_path_original_text}\\{category}"
    path_processed_text = f"{root_path_processed_text}\\{category}"
    for filename in listdir(path_original_text):
        with open(f"{path_original_text}\\{filename}") as file:
            text = cleaner.Cleaner(file.read())
            with open(f"{path_processed_text}\\{filename}", "w") as new_file:
                new_file.write(text.print_final_text())
print('\\' * 100 + "\n")


# PROGRAM DURATION INFORMATION
# ======================================================================================================================
end = time.time()

title = "PROGRAM DURATION INFORMATION"
body = f"Exectution duration: {round(end - start, 2)} seconds'"
debug.information_block(title, body)
# ======================================================================================================================

# https://pythonprogramming.net/text-classification-nltk-tutorial/