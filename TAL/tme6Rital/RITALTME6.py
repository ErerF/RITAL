import numpy as np
import codecs
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from nltk.corpus import stopwords
from sklearn import metrics

# 1 travailler sur le corpus
# 2 Voir dans les classifieurs l'option dans les cas où les donnes sont unbalanced
# 3 Travailler sur le resultat, travail sur la REC/courbe ROC
list_stopwords = stopwords.words('french')


fname = './corpus.tache1.learn.utf8'


alltxts = []
labs = []
s=codecs.open(fname, 'r','utf-8') # pour régler le codage

cpt = 0

alltxts = []
nblignes = 1000
for i in range(nblignes):
    txt = s.readline()
    #print txt

    lab = re.sub(r"<[0-9]*:[0-9]*:(.)>.*","\\1",txt)
    txt = re.sub(r"<[0-9]*:[0-9]*:.>(.*)","\\1",txt)

    #assert(lab == "C" or lab == "M")

    if lab.count('M') >0:
        labs.append(-1)
    else:
        labs.append(1)
    alltxts.append(txt)

    cpt += 1
    if cpt %1000 ==0:
        print(cpt)



alltxts = np.array(alltxts)
labs = np.array(labs)



vectorizer = CountVectorizer(strip_accents='unicode',stop_words=list_stopwords)
BOW = vectorizer.fit_transform(alltxts)
print(vectorizer.get_feature_names())
tfidfMatrix = TfidfTransformer().fit_transform(BOW)


clfSVM = svm.LinearSVC()
clfSVM.fit(BOW,labs)


clfNB = MultinomialNB()
clfNB.fit(BOW,labs)

print(labs)

cv = StratifiedKFold(n_splits=5)

scores = cross_val_score(clfSVM, BOW, labs, cv=cv,scoring='f1')
print("CV SVM")
print(scores)

scores = cross_val_score(clfNB, BOW, labs, cv=cv,scoring='f1')
print("CV")
print(scores)