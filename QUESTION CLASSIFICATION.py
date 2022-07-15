from nltk.corpus import stopwords     # nltk-natural language toolhit
from nltk.tokenize import word_tokenize

#CH-9
data = open("X:/mano/smillarity questions/NCERT/ch-9.txt",'r',errors='ignore')
a = data.read()
# print(a)

#TOKENIZATION
tok9 = word_tokenize(a)
# print(tok9)
print("token for ch 9 : ",len(tok9))

#STOP-WORDS REMOVAL
sw = stopwords.words('english')
sw9 = {w for w in tok9 if not w in sw}
# print(sw9)
print("Token after removal of stop-words in ch 9 : ",len(sw9),'\n')


#CH-10
data = open("X:/mano/smillarity questions/NCERT/ch-10.txt",'r',errors='ignore')
b = data.read()
# print(b)

#TOKENIZATION
tok10 = word_tokenize(b)
# tok10
print("token for ch 10 : ",len(tok10))

#STOP-WORDS REMOVAL
sw10 = {w for w in tok9 if not w in sw}
# sw10
print("Token after removal of stop-words in ch 10 : ",len(sw10),'\n')


#CH-11
data = open("X:/mano/smillarity questions/NCERT/ch-11.txt",'r',errors='ignore')
c = data.read()
# print(c)

#TOKENIZATION
tok11 = word_tokenize(c)
# tok11
print("token for ch 11 : ",len(tok11))

#STOP-WORDS REMOVAL
sw11 = {w for w in tok11 if not w in sw}
# sw11
print("Token after removal of stop-words in ch 11 : ",len(sw11))


inp = " Define Second law of thermodynamics Reversible and irreversible processes"
toki = word_tokenize(inp)
swi = {w for w in toki if not w in sw}
# print(swi)


l9 = []
l10 = []
l11 = []

a = 0
b = 0
for w in sw9:
    if w in swi:
        l9.append(1)
    else:
        l9.append(0)
for x in l9:
    if l9[x]==0:
        a = a+1
    else:
        b = b+1
zero9 = l9.count(0)
one9 = l9.count(1)

print(one9)
print(zero9,'\n')


for w in sw10:
    if w in swi:
        l10.append(1)
    else:
        l10.append(0)
for x in l10:
    if l10[x]==0:
        a = a+1
    else:
        b = b+1
zero10 = l10.count(0)
one10 = l10.count(1)

print(one10)
print(zero10,'\n')


for w in sw11:
    if w in swi:
        l11.append(1)
    else:
        l11.append(0)
for x in l11:
    if l11[x]==0:
        a = a+1
    else:
        b = b+1
zero11 = l11.count(0)
one11 = l11.count(1)

print(one11)
print(zero11)
print(one11+zero11)
