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
        weights = copy.deepcopy(self.index.getTfsForDoc(idDoc))
        for t in dico:
            if t not in weights.keys():
                weights[t] = 0
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
        weights = copy.deepcopy(self.index.getTfsForDoc(idDoc))
        for t in dico:
            if t not in weights.keys():
                weights[t] = 0
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
        weights = copy.deepcopy(self.index.getTfsForDoc(idDoc))
        for t in dico:
            if t not in weights.keys():
                weights[t] = 0
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
    #renvoie un dico (doc : score)
    def getScores(query):
        raise NotImplementedError()
        
    def getRanking(query):
        couples = dict.items().sort(key=itemgetter(1), reverse= True)
        return couples
    
class Vectoriel(IRModel):
    
    def __init__(self,weighter,index):
        self.weighter = weighter
        self.normalized = False
        self.__init__(index)
        
    def getScores(query):
        return
    
class ModeleLangue(IRModel):
        
    def getScores(query):
        return
    
class Okapi(IRModel):
        
    def getScores(query):
        return