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
            
    
    def TfsForStem(self,stem):
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