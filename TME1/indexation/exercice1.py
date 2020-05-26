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
print("TME2")
"""
Q1.1 : Pour le modèle booleen on peut utiliser l'index pour verifier 
si les termes de la requete sont dans le document ou non

Pour le modele vectoriel, on peut utiliser l'index inversé pour avoir le dico
et l'index normal pour avoir les documents
"""



       
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
    scores = sorted(scores.items(),key=lambda item: item[1], reverse= True)
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
        
     #Score de chaque document   
    for docNum, doc in index.items():
        docVecteur = []
        for t in dico:
            if t in doc.keys():
                docVecteur.append(doc[t])
            else:
                docVecteur.append(0)        
        scores[str(docNum)] = np.dot(requeteVecteur,docVecteur) 
    scores = sorted(scores.items(),key=lambda item: item[1], reverse= True)
    return scores
      
  
print(modeleVectoriel(requete,"corpusindex.txt","corpusindexinverse.txt"))


# PAs compris le produit cartésien alors que dans le cours on parle de produit scalaire ou cosinus


#EXERCICE 1 TME 3

#1.1

print("TME3")

requete1 = "top sales"
pertinent1 = [0,1]
requete2 = "sales increase july"
pertinent2 = [2,3]
requete3 = "new home"
pertinent3 = []


print(modeleBooleen(requete1,"corpusindex.txt"))
print(modeleBooleen(requete2,"corpusindex.txt"))
print(modeleBooleen(requete3,"corpusindex.txt"))
print()
print(modeleVectoriel(requete1,"corpusindex.txt","corpusindexinverse.txt"))
print(modeleVectoriel(requete2,"corpusindex.txt","corpusindexinverse.txt"))
print(modeleVectoriel(requete3,"corpusindex.txt","corpusindexinverse.txt"))

#Nous avons considéré que si le score était de 0 alors le doc n'était pas pertinent pour le score
def pa2(res,pertinent):
    res = dict(res[0:2])
    tp = 0
    fp = 0
    for doc,r in res.items():
        if r > 0 and doc in pertinent:
            tp += 1
        if r > 0 and doc not in pertinent:
            fp += 1
    if tp + fp ==0:
        return 0
    return tp/(tp + fp)

    
def ra2(res,pertinent):
    res = dict(res[0:2])
    tp = 0
    fn = 0
    for doc,r in res.items():
        if r > 0 and doc in pertinent:
            tp += 1
        
    for d in pertinent:
        if d not in res.keys() or res[str(d)] == 0:
            fn += 1
    if tp + fn == 0:
        return 0
    return tp/(tp + fn)
    
#Pas d'indication donc on a considéré la f1 mesure
def fa2(res,pertinent):
    P = pa2(res,pertinent)
    R = ra2(res,pertinent)
    if (P+R) == 0:
        return 0
    return 2*(P*R)/(P+R)

listPrecision = []
listPrecision.append(pa2(modeleBooleen(requete1,"corpusindex.txt"),pertinent1))
listPrecision.append(pa2(modeleBooleen(requete2,"corpusindex.txt"),pertinent2))
listPrecision.append(pa2(modeleBooleen(requete3,"corpusindex.txt"),pertinent3))
print(listPrecision)
print(np.mean(np.array(listPrecision)))


listRappel = []
listRappel.append(ra2(modeleBooleen(requete1,"corpusindex.txt"),pertinent1))
listRappel.append(ra2(modeleBooleen(requete2,"corpusindex.txt"),pertinent2))
listRappel.append(ra2(modeleBooleen(requete3,"corpusindex.txt"),pertinent3))
print(listRappel)
print(np.mean(np.array(listRappel)))

listF = []
listF.append(fa2(modeleBooleen(requete1,"corpusindex.txt"),pertinent1))
listF.append(fa2(modeleBooleen(requete2,"corpusindex.txt"),pertinent2))
listF.append(fa2(modeleBooleen(requete3,"corpusindex.txt"),pertinent3))
print(listF)
print(np.mean(np.array(listF)))

# Problème au niveau des fonctions de scores qui sont mauvaises ou mauvaise interprétation 
# de la question ?
#1.2

def NDCG(self,liste,query):
    IDCG = 1
    for i in range(1,len(query.listRelevantDocs)):
        IDCG += 1/math.log(i+1,2)
        
    if liste[0] in query.listRelevantDocs:
        DCG = 1
    else:
        DCG = 0
    for i in range(1,len(liste)):
        if liste[i] in query.listRelevantDocs:
            DCG += 1/math.log(i+1,2)
    
    return DCG/IDCG


"""   


print(buildDocumentCollectionRegex("cacmShort-good.txt"))



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