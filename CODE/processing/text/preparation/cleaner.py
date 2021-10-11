# -*- coding: utf-8 -*-

import pandas as pd

from os import listdir

from processing.text.preparation import tag_names_translation
from utilities import debug

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# from nltk.tokenize import sent_tokenize
# from nltk.tokenize import PunktSentenceTokenizer

from nltk.stem import PorterStemmer

import nltk.tag

class Cleaner():
    
    def __init__(self, text):
        self.text, self.text2 = [text]*2

        # DEBUG OPTIONS
        # ======================================================================================================================
        DEBUG_ALL = False

        FULL_TEXT_DEBUG = False or DEBUG_ALL
        LOWERING_NON_ALPHABETIC_DEBUG = False or DEBUG_ALL
        STOP_WORDS_DEBUG = False or DEBUG_ALL
        STEMMING_DEBUG = False or DEBUG_ALL
        TOKEN_TO_TEXT_DEBUG = False or DEBUG_ALL
        # ======================================================================================================================


        word_tokens = word_tokenize(self.text2)

        # LOWERING AND REMOVING NON-ALPHABETIC TOKENS/WORDS
        # ======================================================================================================================
        word_tokens2 = word_tokens
        word_tokens = [word.lower() for word in word_tokens if word.isalpha()]
        if LOWERING_NON_ALPHABETIC_DEBUG:
            title = "LOWERING AND REMOVING NON-ALPHABETIC TOKENS/WORDS"
            body = []
            body.append(f"Words pre-processed: {word_tokens2}\n")
            body.append(f"Words post-processed: {word_tokens}\n")
            body.append(f"Length Words pre-processed: {len(word_tokens2)}\n")
            body.append(f"Length Words post-processed: {len(word_tokens)}\n")
            debug.information_block(title, *body)
        # ======================================================================================================================

        # STOP WORDS
        # ======================================================================================================================
        # from nltk.corpus import stopwords
        # from nltk.tokenize import word_tokenize

        # from nltk.tokenize import PunktSentenceTokenizer

        STOP_WORDS = set(stopwords.words('english'))

        preprocessed = word_tokens
        word_tokens = [w for w in word_tokens if not w in STOP_WORDS]
        if STOP_WORDS_DEBUG:
            title = "STOP WORDS"
            body = []
            body.append(f"Words pre-processed: {preprocessed}\n")
            body.append(f"Words post-processed: {word_tokens}\n")
            body.append(f"Length Words pre-processed: {len(preprocessed)}\n")
            body.append(f"Length Words post-processed: {len(word_tokens)}\n")
            debug.information_block(title, *body)
        # ======================================================================================================================

        # TAGGING
        # ======================================================================================================================
        tagged = nltk.pos_tag(word_tokens)
        tagged = tag_names_translation.pos_tag_wordnet(tagged)
        # ======================================================================================================================

        # LEMMATISING
        # stemming (mucho mas rapido) vs lemmatization --> canviar a lemmanntiztion es mejor, da la raiz (stem) de la palabra correcta una palabra que existe siempre
        # ======================================================================================================================
        # from nltk.stem import PorterStemmer
        # from nltk.tokenize import sent_tokenize, word_tokenize

        from nltk import WordNetLemmatizer
        lemmatiser = WordNetLemmatizer()

        lemmatised_words = []
        for w, t in tagged: lemmatised_words.append(lemmatiser.lemmatize(w, pos=t))

        preprocessed = word_tokens
        word_tokens = lemmatised_words

        if STEMMING_DEBUG:
            title = "STEMMING"
            body = []
            body.append(f"Words pre-processed: {preprocessed}")
            body.append(f"Words post-processed: {word_tokens}")
            body.append(f"Length Words pre-processed: {len(preprocessed)}")
            body.append(f"Length Words post-processed: {len(word_tokens)}")
            debug.information_block(title, *body)
        # ======================================================================================================================


        # TOKEN TO TEXT
        # ======================================================================================================================
        preprocessed = word_tokens
        # print('preprocessed: ', preprocessed)
        final_text = " ".join(word_tokens)
        # print('final_text: ', final_text)
        #
        if TOKEN_TO_TEXT_DEBUG:
            title = "TOKEN TO TEXT"
            body = []
            body.append(f"Words pre-processed: {preprocessed}")
            body.append(f"Words post-processed: {final_text}")
            body.append(f"Length Words pre-processed: {len(preprocessed)}")
            body.append(f"Length Words post-processed: {len(final_text)}")
            debug.information_block(title, *body)
        # ======================================================================================================================


        # PRINT FULL TEXT (PRE-PROCESSED & POST-PROCESSED)
        # ======================================================================================================================
        if FULL_TEXT_DEBUG:
            print(f"\nRAW TEXT (Words: {len(self.text)})\n{'='*100}\n{self.text}\n{'='*100}")
            print()
            print(f"\nPROCESSED TEXT (Words: {len(self.text2)})\n{'='*100}\n{self.text2}\n{'='*100}")
        # ======================================================================================================================

        self.final_text = final_text

    def print_final_text(self):
        return self.final_text