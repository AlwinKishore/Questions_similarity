# importing the necessary libraries
import pandas as pd
import numpy as np
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# sw contains the list of stopwords
sw = stopwords.words('english')

# get the input (questions)
question1 = str(input("Enter the first question: \n"))
question2 = str(input("Enter the second question: \n"))
# print(question1)
# print(question2)


# pre-processing
# punctuation removal 
# word conversion
# tokenization
# stop word removal 
import re

def text_to_wordlist(text, remove_stop_words=True, stem_words=False):
    # Clean the text, with the option to remove stop_words

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " america ", text)
    text = re.sub(r" USA ", " america ", text)
    text = re.sub(r" u s ", " america ", text)
    text = re.sub(r" uk ", " england ", text)
    text = re.sub(r" UK ", " england ", text)
    text = re.sub(r"India", "india", text)
    text = re.sub(r"Switzerland", "switzerland", text)
    text = re.sub(r"China", "china", text)
    text = re.sub(r"Chinese", "chinese", text) 
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"Quora", "quora", text)
    text = re.sub(r" dms ", "direct messages ", text)  
    text = re.sub(r"demonitization", "demonetization", text) 
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text) 
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iPhone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text) 
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"II", "two", text) 
    text = re.sub(r"III", "three", text)
    text = re.sub(r"IV", "four", text)
    text = re.sub(r"V", "five", text) 
    text = re.sub(r"VI", "six", text)
    text = re.sub(r"VII", "seven", text)
    text = re.sub(r"VIII", "eight", text)
    text = re.sub(r"IX", "nine", text)
    text = re.sub(r"2nd", "two", text) 
    text = re.sub(r"3rd", "three", text)
    text = re.sub(r"4th", "four", text)
    text = re.sub(r"5th", "five", text) 
    text = re.sub(r"6th", "six", text)
    text = re.sub(r"7th", "seven", text)
    text = re.sub(r"8th", "eight", text)
    text = re.sub(r"9th", "nine", text)
    text = re.sub(r"the US", "america", text)
    text = re.sub(r"Astrology", "astrology", text)
    text = re.sub(r"Method", "method", text)
    text = re.sub(r"Find", "find", text) 
    text = re.sub(r"Banglore", "banglore", text)
    text = re.sub(r" J K ", " JK ", text)
    
    # add padding to punctuations and special chars, we still need them later
    
    text = re.sub('\$', " dollar ", text)
    text = re.sub('\%', " percent ", text)
    text = re.sub('\&', " and ", text)
    text = re.sub('\*', " asterik ", text)
    text = re.sub('\+', " plus ", text)
    text = re.sub('\@', " at ", text)
    
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation]).lower()
    print(text)

    text = word_tokenize(text)
    print(text)

    l1 = [w for w in text if not w in sw]

    return(l1)

#print(text_to_wordlist(question1))
#print(text_to_wordlist(question2))
q1 = text_to_wordlist(question1)
q2 = text_to_wordlist(question2)

print(q1)
print(q2)

qn_list = [q1,q2]
# print("Question list:",qn_list)


# lemmatization
# from nltk.stem import WordNetLemmatizer
  
# lemmatizer = WordNetLemmatizer()

# q1 = ' '.join(lemmatizer.lemmatize(w) for w in q1)
# print(q1)
# q2 = ' '.join(lemmatizer.lemmatize(w) for w in q2)
# print(q2)

vectorizer = CountVectorizer(ngram_range=(1, 3))
# q1_v = vectorizer.fit('q1').vocabulary_
q1_v =[]
for i in range(0,len(qn_list)):
    k = vectorizer.fit_transform(qn_list[i])
    q1_v.append(k)

print(q1_v)


def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))

print(jaccard_similarity(q1,q2))


from nltk.corpus import wordnet
 
for i in range(0,len(q1)):
    syn1 = wordnet.synsets(q1[i])[0]

print(syn1)

for i in range(0,len(q2)):
    syn1 = wordnet.synsets(q2[i])[0]

print(syn2)

# # Encode the Document
# vector = vectorizer.transform(q1_v)
 
# # Summarizing the Encoded Texts
# print("Encoded Document is:")
# print(vector.toarray())
