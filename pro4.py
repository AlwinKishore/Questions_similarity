import nltk
from nltk.stem import WordNetLemmatizer
#
# # Init the Wordnet Lemmatizer
# lemmatizer = WordNetLemmatizer()
#
# # Lemmatize Single Word
# print(lemmatizer.lemmatize("bats"))
#
# print(lemmatizer.lemmatize("are"))
#
# print(lemmatizer.lemmatize("feet"))

# sentence = "The striped bats are hanging on their feet for best"
#
# # Tokenize: Split the sentence into words
# word_list = nltk.word_tokenize(sentence)
# print(word_list)
# #> ['The', 'striped', 'bats', 'are', 'hanging', 'on', 'their', 'feet', 'for', 'best']
#
# # Lemmatize list of words and join
# lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
# print(lemmatized_output)

from nltk.corpus import wordnet

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


# 1. Init Lemmatizer
lemmatizer = WordNetLemmatizer()

# 2. Lemmatize Single Word with the appropriate POS tag
word = 'feet'
print(lemmatizer.lemmatize(word, get_wordnet_pos(word)))

# 3. Lemmatize a Sentence with the appropriate POS tag
sentence = "The striped bats are hanging on their feet for best"
print([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(sentence)])
#> ['The', 'strip', 'bat', 'be', 'hang', 'on', 'their', 'foot', 'for', 'best']