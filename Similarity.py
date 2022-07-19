# Program to measure the similarity between
# two sentences using cosine similarity.

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# X = input("Enter first string: ").lower()
# Y = input("Enter second string: ").lower()
X = "ravi went to market and buy 4 orangea and 2 appes in total how much fruits did ravi bought"
Y = "ram went to shopping mall and buy 1pant and 5 shirts. how much cloths does ram bought"

# tokenization
X_list = word_tokenize(X)
Y_list = word_tokenize(Y)

# sw contains the list of stopwords
sw = stopwords.words('english')
l1 =[]
l2 =[]

# remove stop words from the string
X_set = {w for w in X_list if not w in sw}
Y_set = {w for w in Y_list if not w in sw}
print(X_set)
print(Y_set)


# form a set containing keywords of both strings
rvector = X_set.union(Y_set)
print(rvector)
for w in rvector:
	if w in X_set: l1.append(1) # create a vector
	else: l1.append(0)
	if w in Y_set: l2.append(1)
	else: l2.append(0)
print(l1)
print(l2)

c = 0

# cosine formula
for i in range(len(rvector)):
		c+= l1[i]*l2[i]
cosine = c / float((sum(l1)*sum(l2))*0.5)

print("similarity: ", cosine)


#define Jaccard Similarity function
def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

#find Jaccard Similarity between the two sets
print("Jaccard", jaccard(X_set,Y_set))











