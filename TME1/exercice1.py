# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 13:19:47 2020

@author: arnau
"""

from collections import Counter
import porter
import numpy as np
from ast import literal_eval
import json

stopwords = set(['the','a','an','on','behind','under','there','in','on'])
# EXERCICE 1 TME 1
def preprocessing_text(s):
    tab = s.lower().split(" ")
    tab = [a for a in tab if a not in stopwords]
    for i in range(len(tab)):
            tab[i] = porter.stem(tab[i])
    res = Counter(tab)
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




# EXERCICE 1 TME 2
"""
Q1.1 : Pour le modèle booleen on peut utiliser l'index pour verifier 
si les termes de la requete sont dans le document ou non

Pour le modele vectoriel, on peut utiliser l'index inversé pour avoir le dico
et l'index normal pour avoir les documents
"""


print(buildDocumentCollectionRegex("cacmShort-good.txt"))
       
requete = "home sales top"
#Fonction qui va vérifier que sur la requete donnnée qui ne correspond qu'à faire des "ET" logiques
def modeleBooleen(requete,filenameIndex):
    requete = preprocessing_text(requete)
    index = literal_eval(open(filenameIndex).read())
    scores = {}
    for docNum, doc in index.items():
        for t in requete:
            if t not in list(doc.keys()):
                scores[str(docNum)] = 0
        if str(docNum) not in scores.keys():
            scores[str(docNum)] = 1
    print(index)
    return scores
    
print(modeleBooleen(requete,"corpusindex.txt"))
  

    
# Prend la requete sous forme de vecteur et realise un vecteur pour chaque document 
# à partir de l'index Inverse
def modeleVectoriel(requete,filenameIndex,filenameIndexRev):
    requete = preprocessing_text(requete)
    indexRev = literal_eval(open(filenameIndexRev).read())
    index = literal_eval(open(filenameIndex).read())
    dico = list(indexRev.keys())
    scores = {}
    requeteVecteur = []
    
    #Vecteur de la requete
    for t in dico:
        if t in requete:
            requeteVecteur.append( requete[t])
        else:
            requeteVecteur.append(0)
    print(requeteVecteur)
        
     #Score de chaque document   
    for docNum, doc in index.items():
        docVecteur = []
        for t in dico:
            if t in doc.keys():
                docVecteur.append(doc[t])
            else:
                docVecteur.append(0)        
        scores[str(docNum)] = np.dot(requeteVecteur,docVecteur) 
    return scores
      
  
print(modeleVectoriel(requete,"corpusindex.txt","corpusindexinverse.txt"))


# PAs compris le produit cartésien alors que dans le cours on parle de produit scalaire ou cosinus


#EXERCICE 1 TME 3
"""   


















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
    collection = {}
    del corpus[0]
    for doc in corpus:
        ident = re.search(r"[0-9]",doc).group(0)
        text = re.search(r"\.T([\s\S]*?)\.[ITBAKWX]",doc).group(1)
        print(text)
        collection[str(ident)] = text
    print(collection)
    file = open("camshort-dict.txt","w")
    file.write(json.dumps(collection)) 
    return 0
    
    
"""