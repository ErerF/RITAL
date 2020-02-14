# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 19:07:19 2020

@author: arnau
"""

import Parsing

class Weighter():
    
    def __init__(self,index):
        self.index = index
        
    # dict( terme : poids) du doc
    def getWeightsForDoc(idDoc):
        raise NotImplementedError()
        
    # dict (doc : poids) du stem
    def getWeightsForStem(stem):
        raise NotImplementedError()
                
    # dict (terme : poids) de la query
    def getWeightsForQuery(query):
        raise NotImplementedError()
        
class Weighter1(Weighter):
    
    def getWeightsForDoc(idDoc):
        print(self.index.getTfsForDoc(idDoc))
        tfs = self.index.getTfsForDoc(idDoc)
        for k,v in tfs.items():
            v = 0
        print(self.index.getTfsForDoc(idDoc))
    def getWeightsForStem(stem):
        return
    def getWeightsForQuery(query):
        return
w = Weighter1()
    
class IRModel():
    
    def getScores(query):
        raise NotImplementedError()
        
    def getRanking(query):
        return