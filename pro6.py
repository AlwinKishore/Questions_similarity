# WuPalmer – WordNet Similarity

from nltk.corpus import wordnet

syn1 = wordnet.synsets('hello')[0]
syn2 = wordnet.synsets('selling')[0]

print("hello name :  ", syn1.name())
print("selling name :  ", syn2.name())