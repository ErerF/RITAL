# -*- coding: utf-8 -*-
"""
Created on Wed May 20 04:15:44 2020

@author: arnau
"""

import Parsing

a = np.array()



class Query():
    
    def __init__(self):
        self.id = 0
        self.text = 0
        self.listRelevantDocs = []
        
class QueryParser():
    
    def __init__(self):
        self.Queries = []
        
    def parseQueries(path):
        return self.Queries
    


class EvalMesure():
    
    def evalQuery(liste,query):
        raise NotImplementedError()
        
        