from nltk.corpus import wordnet

wordnet_map = {
    "N": wordnet.NOUN,
    "V": wordnet.VERB,
    "J": wordnet.ADJ,
    "R": wordnet.ADV
}


def pos_tag_wordnet(tagged_text):
    """
        Create pos_tag with wordnet format
    """

    # map the pos tagging output with wordnet output
    pos_tagged_text = [
        (word, wordnet_map.get(pos_tag[0])) if pos_tag[0] in wordnet_map.keys()
        else (word, wordnet.NOUN)
        for (word, pos_tag) in tagged_text
    ]

    return pos_tagged_text

    # https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python