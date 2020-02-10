# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 01:10:36 2020

@author: arnau
"""

from collections import Counter

import numpy as np
from ast import literal_eval
import json

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
            d.I = re.search(r"[0-9]",doc).group(0)
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
        
    def getTfsForDoc():
        
    def getTfIDFsForDoc():
        
    def TfsForStem():
        
    def getTfIDFsForStem():
        
    def getStrDoc():
