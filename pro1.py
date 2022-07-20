import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import sentence_transformers

df = pd.read_csv("D:/PG/internship/Project/dataset/questions.csv", index_col=0)
# print(df.head(10))

# find the dimension of the dataset
# print(df.shape)

# data types of the dataset
# print(df.dtypes)

# columns in the dataset
# print("Column Names: ",df.columns)

# print the number of duplicates in the dataset
# print("Count of Duplicated values: ",df.duplicated().sum())

# # printing the first question in question1
# qn1 = df['question1'][0]
# print(qn1)
#
# # printing the first question in question2
# qn2 = df['question2'][0]
# print(qn2)
#
# # tokenization of questions
# # tokenization qn1
# qn1_list = word_tokenize(qn1)
# print(qn1_list)
#
# # tokenization qn2
# qn2_list = word_tokenize(qn2)
# print(qn2_list)
#
# # sw contains the list of stopwords
sw = stopwords.words('english')


#
# # remove stop words from the string qn1
# qn1_set = {w for w in qn1_list if not w in sw}
# print(qn1_set)
#
# # remove stop words from the string qn2
# qn2_set = {w for w in qn2_list if not w in sw}
# print(qn2_set)
#
#
# # form a set containing keywords of both strings
def check(qn1, qn2):
    l1 = []
    l2 = []
    rvector = qn1.union(qn2)
    # print(rvector)
    for w in rvector:
        if w in qn1:
            l1.append(1)  # create a vector
        else:
            l1.append(0)
        if w in qn2:
            l2.append(1)
        else:
            l2.append(0)
    print(l1)
    print(l2)
    c = 0
    # cosine formula
    for i in range(len(rvector)):
        c += l1[i] * l2[i]
    cosine = c / float((sum(l1) * sum(l2)) * 0.5)

    print("Cosine : %.3f" % cosine)

    #   jaccard similarity
    intersection = len(list(set(l1).intersection(l2)))
    union = (len(l1) + len(l2)) - intersection
    print("Jaccard : %.4f" % (float(intersection) / union))
    return


#
# check(qn1_set,qn2_set)


for i in range(0, 10, 1):
    qn1 = df['question1'][i].lower()
    qn2 = df['question2'][i].lower()
    qn1_list = word_tokenize(qn1)
    qn2_list = word_tokenize(qn2)
    qn1_set = {w for w in qn1_list if not w in sw}
    qn2_set = {w for w in qn2_list if not w in sw}
    # print(qn1,qn2)
    print('Question set: ', i)
    check(qn1_set, qn2_set)
