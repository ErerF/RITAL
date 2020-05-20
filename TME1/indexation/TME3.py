# -*- coding: utf-8 -*-
"""
Created on Wed May 20 04:15:44 2020

@author: arnau
"""

#import cisi

import math


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
        return rappel/len(query.listRelevantDocs)
        
class Fmesure(EvalMesure):
    
    def __init__(self,rang):
        self.rang = rang
        
    def evalQuery(liste,query):
        if len(liste) > self.rang:
            liste = liste[:rang-1]
            
        evalPrecision = Precision(self.rang)
        P = evalPrecision.evalQuery(liste,query)
        evalRappel = Rappel(self.rang)
        R = evalRappel.evalQuery(liste,query)

        return 2*P*R/(P+R)
    
class PrecisionMoyenne(EvalMesure):
    """
    Somme des produits Precision à rang k fois le document au rang k est pertinent ou non
    Divisé par le nombre de docs pertinents à la query
    """
    def evalQuery(liste,query):
        numberRelevantDocs = len(query.listRelevantDocs)
        avgP = 0
        
        for i in range(len(liste)):
            evalPrecision = Precision(i)
            pak = evalPrecision(liste,query)
            
            if liste[i] in query.listRelevantDocs:
                Rdkq = 1
            else:
                Rdkq = 0
            
            avgP += Rdkq * pak  
            
        return avgP/numberRelevantDocs
    
class ReciprocalRank(EvalMesure):
    """
    Dans le cours il n'y a que le mean reciproqual rank mais cela ne semble pas correspondre
    à ce qui est demandé là. Je suppose qu'il faut renvoyer le rang du premier doc pertinent 
    de la liste
    """
    def evalQuery(liste,query):
        for i in range(len(liste)):
            if liste[i] in query.listRelevantDocs:
                return 1/(i+1)
        return 1/len(liste)
    

class NDCG(EvalMesure):
    """
    Pas d'info sur quels documents sont les plus pertinents  parmis ceux jugés comme
    pertinents je crois, donc on met à 1 le reli de la formule DCG
    """
    def evalQuery(liste,query):
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
        
        
class EvalIRModel():
    
    