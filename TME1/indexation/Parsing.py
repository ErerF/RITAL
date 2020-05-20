# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 01:10:36 2020

@author: arnau
"""

from collections import Counter

import numpy as np
from ast import literal_eval
import json
from TextRep import *
import porter
import re
import copy

class Document():
    
    def __init__(self):
        '''
        Constructor
        '''
        self.I = "e"
        self.T = "e"
        self.B = "e"
        self.A = "e"
        self.K = "e"
        self.W = "e"
        self.X = "e"
    
    def __repr__(self):
        return self.I + " : " + self.T 
        
class Parser():
    
    def __init__(self):
        self.collection = dict()
        
    def parsing(self,filename):
        corpus = open(filename).read().split(".I")
        del corpus[0]
        for doc in corpus:
            d = Document()
            d.I = re.search(r"[0-9]+",doc).group(0)
            d.T = self.getElement("T",doc)
            d.B = self.getElement("B",doc)
            d.A = self.getElement("A",doc)
            d.K = self.getElement("K",doc)
            d.X = self.getElement("X",doc)
               
            self.collection[str(d.I)] = d
    
    def getElement(self,pattern,doc):
        res = re.search(r"\." + pattern + "([\s\S]*?)\.[ITBAKWX]",doc)
        if isinstance(res,re.Match):
            return res.group(1)
        return "e"
    
    
p = Parser()
p.parsing("../cacmShort-good.txt")
print(p.collection)


class IndexerSimple():
    
    def __init__(self):
        self.index = dict()
        self.indexInverse = dict()
    
    def indexation(self,collection):
        #on prend une collection (un parser) de Documents et on associe a chaque
        # id de document la liste de ses mots (seulement le texte ou aussi tout le reste ?)

        stemmer = PorterStemmer()
        #manque des docs du texte, bizarre ...
        for key,doc in collection.items():
            normalizedDoc = stemmer.getTextRepresentation(doc.T)
            self.index[str(doc.I)] = normalizedDoc
            
            for word,occurences in normalizedDoc.items():
                if word in self.indexInverse.keys():
                    if str(key) in self.indexInverse[word]:
                        self.indexInverse[word][str(key)] = indexInverse[word][str(key)] + normalizedDoc[word]
                    else:
                        self.indexInverse[word][str(key)] = normalizedDoc[word]
            
                else:
                    self.indexInverse[word] = dict()
                    self.indexInverse[word][str(key)] = normalizedDoc[word]
            
        
    def getTfsForDoc(self,ident):
        return self.index[str(ident)]
    
    def getTfIDFsForDoc(self,ident):
        N = len(self.index)
        doc = copy.deepcopy(self.index[str(ident)])
        for k,v in self.index[str(ident)].items():
            doc[k] = self.index[str(ident)][k] * np.log((1+N) /(1+ len(self.indexInverse[k])))
        return doc
            
    
    def getTfsForStem(self,stem):
        return self.indexInverse[stem]
    
    def getTfIDFsForStem(self,stem):
        N = len(self.index)
        stem_tfidf = copy.deepcopy(self.indexInverse[stem])
        for k,v in self.indexInverse[stem].items():
            stem_tfidf[k] = self.index[str(k)][stem] * np.log((1+N)/(1 + len(self.indexInverse[stem])))
        return stem_tfidf
    
    def getStrDoc(self,parser,ident):
        return parser.collection[str(ident)].T
        
    
ind = IndexerSimple()
ind.indexation(p.collection)

print(ind.getTfIDFsForDoc(2))
print(ind.getStrDoc(p,2))
print(ind.getTfsForStem("extract"))

class Weighter():
    
    def __init__(self,index):
        self.index = index
        
    # dict( terme : poids) du doc
    def getWeightsForDoc(self,idDoc):
        raise NotImplementedError()
        
    # dict (doc : poids) du stem
    def getWeightsForStem(self,stem):
        raise NotImplementedError()
                
    # dict (terme : poids) de la query
    # normaliser la requete
    def getWeightsForQuery(self,query):
        raise NotImplementedError()
        
class Weighter1(Weighter):
    
    def getWeightsForDoc(self,idDoc):
        dico = self.index.indexInverse.keys()
        weights = {}
        copyWeights = copy.deepcopy(self.index.getTfsForDoc(idDoc))
        for t in dico:
            if t not in copyWeights.keys():
                weights[t] = 0
            else:
                weights[t] = copyWeights[t]
        return weights
        
    def getWeightsForStem(self,stem):
        docsNb = self.index.index.keys()
        weights = copy.deepcopy(self.index.getTfsForStem(stem))
        for d in docsNb:
            if d not in weights.keys():
                weights[d] = 0
        return weights
    
    def getWeightsForQuery(self,query):
        stemmer = PorterStemmer()
        weights = {}
        terms = list(Counter(stemmer.getTextRepresentation(query)).elements())
        dico = self.index.indexInverse.keys()
        for t in dico:
            if t in terms:
                weights[t] = 1
            else:
                weights[t] = 0
        return weights
    
w = Weighter1(ind)
print(1)
print(w.getWeightsForDoc(2))
print(w.getWeightsForStem('extract'))
print(w.getWeightsForQuery("home glossary use proposal report report "))

class Weighter2(Weighter):
    
    def getWeightsForDoc(self,idDoc):
        dico = self.index.indexInverse.keys()
        weights = {}
        copyWeights = copy.deepcopy(self.index.getTfsForDoc(idDoc))
        for t in dico:
            if t not in copyWeights.keys():
                weights[t] = 0
            else:
                weights[t] = copyWeights[t]
        return weights
        
        
    def getWeightsForStem(self,stem):
        docsNb = self.index.index.keys()
        weights = copy.deepcopy(self.index.getTfsForStem(stem))
        for d in docsNb:
            if d not in weights.keys():
                weights[d] = 0
        return weights
    
    def getWeightsForQuery(self,query):
        stemmer = PorterStemmer()
        weights = {}
        terms = list(Counter(stemmer.getTextRepresentation(query)).elements())
        count = Counter(terms)
        dico = self.index.indexInverse.keys()
        for t in dico:
            weights[t] = count[t]
        return weights
    
w = Weighter2(ind)
print(2)
print(w.getWeightsForDoc(2))
print(w.getWeightsForStem('extract'))
print(w.getWeightsForQuery("home glossary use proposal report report"))


class Weighter3(Weighter):
    
    def getWeightsForDoc(self,idDoc):
        dico = self.index.indexInverse.keys()
        weights = {}
        copyWeights = copy.deepcopy(self.index.getTfsForDoc(idDoc))
        for t in dico:
            if t not in copyWeights.keys():
                weights[t] = 0
            else:
                weights[t] = copyWeights[t]
        return weights
        
    def getWeightsForStem(self,stem):
        docsNb = self.index.index.keys()
        weights = copy.deepcopy(self.index.getTfsForStem(stem))
        for d in docsNb:
            if d not in weights.keys():
                weights[d] = 0
        return weights
    
    def getWeightsForQuery(self,query):
        N = len(self.index.index)
        stemmer = PorterStemmer()
        weights = {}
        terms = list(Counter(stemmer.getTextRepresentation(query)).elements())
        dico = self.index.indexInverse.keys()
        indexInv = self.index.indexInverse
        for t in dico:
            if t in terms:
                weights[t] = np.log((1 + N)/(1 * len(indexInv[t]) ))
            else:
                weights[t] = 0
        return weights
    
w = Weighter3(ind)
print(3)
print(w.getWeightsForDoc(2))
print(w.getWeightsForStem('extract'))
print(w.getWeightsForQuery("home glossary use proposal report report "))

class Weighter4(Weighter):
    
    def getWeightsForDoc(self,idDoc):
        dico = self.index.indexInverse.keys()
        tfsForDoc = copy.deepcopy(self.index.getTfsForDoc(idDoc))
        weights = {}
        for term in dico:
            if term in tfsForDoc.keys():
                weights[term] = 1 + np.log(tfsForDoc[term])
            else:
                weights[term] = 0
        return weights
        
    def getWeightsForStem(self,stem):
        docsNb = self.index.index.keys()
        tfsForStem = copy.deepcopy(self.index.getTfsForStem(stem))
        weights = {}
        for d in docsNb:
            if d in tfsForStem.keys():
                weights[d] = 1 + np.log(tfsForStem[d])
            else:
                weights[d] = 0
        return weights
    
    def getWeightsForQuery(self,query):
        N = len(self.index.index)
        stemmer = PorterStemmer()
        weights = {}
        terms = list(Counter(stemmer.getTextRepresentation(query)).elements())
        dico = self.index.indexInverse.keys()
        indexInv = self.index.indexInverse
        for t in dico:
            if t in terms:
                weights[t] = np.log((1 + N)/(1 * len(indexInv[t]) ))
            else:
                weights[t] = 0
        return weights
    
w = Weighter4(ind)
print(4)
print(w.getWeightsForDoc(2))
print(w.getWeightsForStem('extract'))
print(w.getWeightsForQuery("home glossary use proposal report report "))

class Weighter5(Weighter):
    
    def getWeightsForDoc(self,idDoc):
        N = len(self.index.index)
        dico = self.index.indexInverse.keys()
        tfsForDoc = copy.deepcopy(self.index.getTfsForDoc(idDoc))
        weights = {}
        indexInv = self.index.indexInverse
        
        for term in dico:
            if term in tfsForDoc.keys():
                weights[term] = (1 + np.log(tfsForDoc[term])) *  np.log((1 + N)/(1 * len(indexInv[term]) ))
            else:
                weights[term] = 0
        return weights
        
    def getWeightsForStem(self,stem):
        N = len(self.index.index)
        indexInv = self.index.indexInverse
        docsNb = self.index.index.keys()
        tfsForStem = copy.deepcopy(self.index.getTfsForStem(stem))
        weights = {}
        for d in docsNb:
            if d in tfsForStem.keys():
                weights[d] = 1 + np.log(tfsForStem[d]) *  np.log((1 + N)/(1 * len(indexInv[stem]) ))
            else:
                weights[d] = 0
        return weights
    0
    def getWeightsForQuery(self,query):
        N = len(self.index.index)
        stemmer = PorterStemmer()
        weights = {}
        terms = list(Counter(stemmer.getTextRepresentation(query)).elements())
        count = Counter(terms)
        dico = self.index.indexInverse.keys()
        indexInv = self.index.indexInverse
        for t in dico:
            if t in terms:
                weights[t] = (1 + np.log(count[t])) * np.log((1 + N)/(1 * len(indexInv[t]) ))
            else:
                weights[t] = 0
        return weights
    
w = Weighter5(ind)
print(5)
print(w.getWeightsForDoc(2))
print(w.getWeightsForStem('extract'))
print(w.getWeightsForQuery("home glossary use proposal report report "))
    
class IRModel():
    
    def __init__(self,index):
        self.index = index
        self.scores = {}
        self.couples = []
        
        self.docWeights = {} # couple document/representation du doc
        self.setDocWeights() 
        print(self.docWeights)
        
    def setDocWeights(self):
        '''
        Transforme les documents en leurs représentations selon le weighter choisi
        '''
        for k,v in self.index.index.items():
            self.docWeights[str(k)] = list(self.weighter.getWeightsForDoc(k).values())
        
    #renvoie un dico (doc : score)
    def getScores(self,query):
        raise NotImplementedError()
        
    def getRanking(self):
        couples = sorted(self.scores.items(),key=lambda item: item[1], reverse= True)
        return couples
    
class Vectoriel(IRModel):
    
    def __init__(self,weighter,index,normalized):
        print(index)
        self.weighter = weighter
        self.normalized = normalized
        IRModel.__init__(self,index)
        

        print()
        print()
        print()
        self.norms = {} # couples  doc/norme euclidienne
        self.setNorms()
        print(self.norms)
    

        
        
    def setNorms(self):
        '''
        Calcule la norme de chaque representation de document
        Fait une fois lors de la creation de l'instance de Vectoriel
        '''
        for k,v in self.docWeights.items():
            self.norms[str(k)] = np.linalg.norm(np.array(v))

        
    def getScores(self,query):
        queryWeights = list(self.weighter.getWeightsForQuery(query).values())
        print("queryWeights")
        print(queryWeights)
        dico = self.index.indexInverse.keys()
        if self.normalized:
            print("Cosinus")
            queryNorm = np.linalg.norm(np.array(queryWeights))
            for docNum,docRep in self.docWeights.items():
                self.scores[str(docNum)] = np.dot(docRep,queryWeights)/(queryNorm*self.norms[str(docNum)])
        else:
            print("Scalaire")
            for docNum,docRep in self.docWeights.items():
                self.scores[str(docNum)] = np.dot(docRep,queryWeights)
        return self.scores
    

w = Weighter1(ind)
print(1)
print("representation du document")
print(w.getWeightsForDoc(9))
print("poids du stem")
print(w.getWeightsForStem('extract'))
print("representation de la query")
print(w.getWeightsForQuery("home glossary use proposal report report "))

vectoriel = Vectoriel(w,ind,True)
print(vectoriel.getScores("home glossary use proposal report report "))
print(vectoriel.getRanking())

    
class ModeleLangue(IRModel):
        
    def __init__(self,weighter,index):
        self.weighter = weighter
        IRModel.__init__(self,index)
        
    def getScores(self,query):
        #entre 0.2 et 0.8
        lamb = 0.8
        print(query)
        queryWeights = self.weighter.getWeightsForQuery(query)
        print(queryWeights)
        queryStems = []
        for k,v in queryWeights.items():
            for i in range(v):
                queryStems.append(k)
        print(queryStems)
        
        tfTotalSurCol = 0
        for doc, v in self.index.index.items():
            tfTotalSurCol = tfTotalSurCol + sum(list(v.values()))
        
        for docNum,docRep in self.docWeights.items():
                 proba = 1
                 for t in queryStems:
                     # pt = tf(t) / sum sur tous les t de tf(t)
                     # modele de langue du document
                     weightsStemDeDoc = self.weighter.getWeightsForDoc(docNum)
                     probaT_ThetaD = weightsStemDeDoc[t]/ sum(list(weightsStemDeDoc.values()))
                     #modele de langue de la collection
                     tfDeTSurCol = sum(list(self.weighter.getWeightsForStem(t).values()))

                     probaT_ThetaC = tfDeTSurCol / tfTotalSurCol
                     
                     proba = proba * (1-lamb)*probaT_ThetaD+ lamb*probaT_ThetaC
                     
                 self.scores[str(docNum)] = proba
            
        return self.scores
    

# Utilisation du TF pour calculer les Pt sur la query avec le weighter 2
w = Weighter2(ind)
print(1)

langue = ModeleLangue(w,ind)
print("ModeleLangue")
print(langue.getScores("home glossary use proposal report report "))
print(langue.getRanking())


class Okapi(IRModel):
        
    def __init__(self,weighter,index):
        self.weighter = weighter
        IRModel.__init__(self,index)
        
    def getScores(self,query):
        k1 = 1.2
        b = 0.75
        
        # taille moyenne d un texte après avoir stem
        avgdl = 0
        for doc, v in self.index.index.items():
            avgdl = avgdl + sum(list(v.values()))
        avgdl = avgdl / len(self.index.index.items())
        
        #Stemmatisation de la query
        stemmer = PorterStemmer()
        queryWeights = stemmer.getTextRepresentation(query)
        queryStems = []
        
        for k,v in queryWeights.items():
            for i in range(v):
                queryStems.append(k)
        print(queryStems)
        
        #Recuperation de l'IDF des termes de la query
        queryTermsIDF = self.weighter.getWeightsForQuery(query)
        print(queryTermsIDF)
        
        # Evaluer le score Okapi BM 25 pour chaque document
        for docNum,docRep in self.docWeights.items():
            score = 0
            docSize = sum(list(self.index.index[str(docNum)].values()))
            tfDoc = self.weighter.getWeightsForDoc(docNum)


            for t in queryStems:
                if t in queryTermsIDF.keys() and t in tfDoc.keys():
                    scoreTemp = queryTermsIDF[t] * ((tfDoc[t] * (k1 + 1))/(tfDoc[t] + k1 * (1 - b + b *(docSize/avgdl))))

                    score = score + scoreTemp
                     
            self.scores[str(docNum)] = score
            
        return self.scores
    
"""
# Besoin de l'idf et du tf pour faire l'okapiBM25 donc weighter3
w = Weighter3(ind)

oka = Okapi(w,ind)
print("OkapiBM25")
print(oka.getScores("home glossary use proposal report report "))
print(oka.getRanking())

print("fin")
print(ind.index)
"""