# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 13:19:47 2020

@author: arnau
"""

from collections import Counter
import porter
import numpy as np
from ast import literal_eval

stopwords = set(['the','a','an','on','behind','under','there','in','on'])
# EXERCICE 1
def preprocessing_text(s):
    tab = s.lower().split(" ")
    tab = [a for a in tab if a not in stopwords]
    for i in range(len(tab)):
            tab[i] = porter.stem(tab[i])
    res = collections.Counter(tab)
    return dict(res)



corpus = "./corpus.txt"

def index(filename):
    lines = open(filename).read().splitlines()
    res = {i : preprocessing_text(l) for i,l in zip(range(len(lines)),lines)}
    indexFile = open("corpusindex.txt","w")
    indexFile.write(str(res))
    indexFile.close()
index(corpus)

def index_inverse(filename):
    lines = open(filename).read().splitlines()
    dic = dict()
    for i in range(len(lines)):
        line = preprocessing_text(lines[i])
        for (k,v) in line.items():
            if k not in dic.keys():
                dic[k] = {}
                dic[k][str(i)] = '1'
            else:
                dic[k][str(i)] = '1'
    indexFile = open("corpusindexinverse.txt","w")
    indexFile.write(str(dic))
    indexFile.close()
    
index_inverse(corpus)

def tf_idf(N):
    index = literal_eval(open("corpusindex.txt").read())
    indexInv =  literal_eval(open("corpusindexinverse.txt").read())
    for i in range(N):
        for (k,v) in index[i].items():
            index[i][k] = index[i][k]*np.log((1+N)/(1 + len(indexInv[k])))
    return index
print("tfidf")
print(tf_idf(4))


# EXERCICE 2
print("EXO 2 ")
def buildDocCollectionSimple(filename):
    lines = open(filename).read().splitlines()
    dic = {}
    for i in range(len(lines)):
        if ".I" in lines[i]:
            ident = lines[i].replace(".I","").replace(" ","")
            dic[ident] = ""
        if ".T" in lines[i]:
            i = i +1
            while((".B" not in lines[i]) and (".A" not in lines[i]) and (".I" not in lines[i])):
                print(i)
                dic[ident] = dic[ident] + lines[i]
                i = i +1
            i = i - 1
    return dic

#sprint(buildDocCollectionSimple("cacmShort-good.txt"))

def buildDocumentCollectionRegex(filename):
    corpus = open(filename).read().split(".I")
    print(corpus[1])
    collection = {}
    print(re.search(r"[0-9]",corpus[1]).group(0))
    print(corpus[1])
    print("aaaaaaaa")
    print(re.search(r"\.T[\s\S]*\.B",corpus[1]).group(0))
    return -1
    
    
print(buildDocumentCollectionRegex("cacmShort-good.txt"))
                 