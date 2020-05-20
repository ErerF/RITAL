# -*- coding: utf-8 -*-
"""
Created on Wed May 20 04:15:44 2020

@author: arnau
"""

#import cisi




class Query():
    
    def __init__(self):
        self.I = "e"
        self.W = "e"
        self.listRelevantDocs = []
        
    def repr(self):
        return self.I + " " + self.W + " " + self.listRelevantDocs
        
class QueryParser():
    
    def __init__(self):
        self.Queries = []
        
    def parseQueries(self,qryFile,relFile):
        
    
        corpus = [x+".I " for x in open(qryFile).read().split(".I ")]
        del corpus[0]
        for query in corpus:
            q = Query()
            q.I = re.search(r"[0-9]+",query).group(0)
            print(q.I)
            q.W = self.getElement("W",query)
            self.Queries.append(q)
            
        jugements = open(relFile).read().splitlines()
        for j in jugements:
            idQuery = re.search(r"([0-9]+)[ ]*([0-9]+)",j).group(1)
            docPertinent = re.search(r"([0-9]+)[ ]*([0-9]+)",j).group(2)
            self.Queries[int(idQuery)-1].listRelevantDocs.append(int(docPertinent))
            
        return self.Queries
    
    def getElement(self,pattern,query):
        res = re.search(r"\." + pattern + "([\s\S]*?)\.[ITBAKWXN]?",query)
        if isinstance(res,re.Match):
            return res.group(1)
        return "e"
    
    
parser = QueryParser()
queries = parser.parseQueries("cisi/cisi.qry","cisi/cisi.rel")
"""for d in queries[0].listRelevantDocs:
    print(d)"""


class EvalMesure():
    
    def evalQuery(liste,query):
        raise NotImplementedError()
        
class Precision(EvalMesure):
    
    def __init__(self,rang):
        self.rang = rang
        
    def evalQuery(liste,query):
        if len(liste) > self.rang:
            liste = liste[:rang-1]
        precision = 0
        for doc in liste:
            if doc in query.listRelevantDocs:
                precision += 1
        return precision/self.rang

class Rappel(EvalMesure):
    
    def __init__(self,rang):
        self.rang = rang
        
    def evalQuery(liste,query):
        if len(liste) > self.rang:
            liste = liste[:rang-1]
        rappel = 0
        for doc in liste:
            if doc in query.listRelevantDocs:
                rappel += 1
        return precision/len(query.listRelevantDocs)
        