# -*- coding: utf-8 -*-
"""
Created on Wed May 20 04:15:44 2020

@author: arnau
"""

import TME1_2_Projet

import math


class Query():
    
    def __init__(self):
        self.I = "e"
        self.W = "e"
        self.listRelevantDocs = []
        
    def repr(self):
        return self.I + " " + self.W + " " + str(self.listRelevantDocs)
        
class QueryParser():
    
    def __init__(self):
        self.Queries = []
        
    def parseQueries(self,qryFile,relFile):
        
    
        corpus = [x+".I " for x in open(qryFile).read().split(".I ")]
        del corpus[0]
        for query in corpus:
            q = Query()
            q.I = re.search(r"[0-9]+",query).group(0)
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
    
    def evalQuery(self,liste,query):
        raise NotImplementedError()
        
class Precision(EvalMesure):
    
    def __init__(self,rang):
        self.rang = rang
        
    def evalQuery(self,liste,query):
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
        
    def evalQuery(self,liste,query):
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
        
    def evalQuery(self,liste,query):
        if len(liste) > self.rang:
            liste = liste[:rang-1]
            
        evalPrecision = Precision(self.rang)
        P = evalPrecision.evalQuery(liste,query)
        evalRappel = Rappel(self.rang)
        R = evalRappel.evalQuery(liste,query)
        print(P)
        print(R)
        return 2*P*R/(P+R)
    
class PrecisionMoyenne(EvalMesure):
    """
    Somme des produits Precision à rang k fois le document au rang k est pertinent ou non
    Divisé par le nombre de docs pertinents à la query
    """
    def evalQuery(self,liste,query):
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
    def evalQuery(self,liste,query):
        for i in range(len(liste)):
            if liste[i] in query.listRelevantDocs:
                return 1/(i+1)
        return 1/len(liste)
    

class NDCG(EvalMesure):
    """
    Pas d'info sur quels documents sont les plus pertinents  parmis ceux jugés comme
    pertinents je crois, donc on met à 1 le reli de la formule DCG
    """
    def evalQuery(self,liste,query):
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
    
    def __init__(self,index,queries):
        self.index = index
        self.queries = queries
    
 
    def evalModel(self, model):
        IRModel = model
        
        rang = 30
        
        listPrecision = []
        listRappel = []
        listFmesure = []
        listReciprocalRank = []
        listNDCG = []
        
        Mprecision = Precision(rang)
        Mrappel = Rappel(rang)
        Mfmesure = Fmesure(rang)
        Mreciprocalrank = ReciprocalRank()
        Mndcg = NDCG()
        
        for q in queries[:2]:
            print(q.repr())
            IRModel.getScores(q.W)
            listeMeilleursDocs = [int(x[0]) for x in IRModel.getRanking()[:rang]]
            print(listeMeilleursDocs)
            listPrecision.append(Mprecision.evalQuery(listeMeilleursDocs,q))
            listRappel.append(Mrappel.evalQuery(listeMeilleursDocs,q))
            listFmesure.append(Mfmesure.evalQuery(listeMeilleursDocs,q))
            listReciprocalRank.append(Mreciprocalrank.evalQuery(listeMeilleursDocs,q))
            listNDCG.append(Mndcg.evalQuery(listeMeilleursDocs,q))
            
            
        listPrecision = np.array(listPrecision)
        listRappel = np.array(listRappel)
        listFmesure = np.array(listFmesure)
        listReciprocalRank = np.array(listReciprocalRank)
        listNDCG = np.array(listNDCG)
        
        print("Precision")
        print(listPrecision)
        print(np.mean(listPrecision))
        print(np.std(listPrecision))
        
        print("Rappel")
        print(listRappel )
        print(np.mean(listRappel ))
        print(np.std(listRappel ))
        
        print("Fmesure")
        print(listFmesure)
        print(np.mean(listFmesure))
        print(np.std(listFmesure))
        
        print("ReciprocalRank")
        print(listReciprocalRank)
        print(np.mean(listReciprocalRank))
        print(np.std(listReciprocalRank))
        
        print("NDCG")
        print(listNDCG)
        print(np.mean(listNDCG))
        print(np.std(listNDCG))
        
        return

    
p = Parser()
p.parsing("cisi/cisi.txt")

ind = IndexerSimple()
ind.indexation(p.collection)

parser = QueryParser()
queries = parser.parseQueries("cisi/cisi.qry","cisi/cisi.rel")



evalIR = EvalIRModel(ind,queries)

#IRMODEL VECTORIEL COSINUS
print("Vectoriel Cosinus")
w = Weighter1(ind)
#evalIR.evalModel(Vectoriel(w,ind,True))

"""

"""

#IRMODEL VECTORIEL SCALAIRE 
print("Vectoriel scalaire")
w = Weighter1(ind)
#evalIR.evalModel(Vectoriel(w,ind,False))

"""

"""

#IRMODEL ModeleLangue 
print("ModeleLangue")
w = Weighter2(ind)
evalIR.evalModel(ModeleLangue(w,ind))

"""

"""

#IRMODEL OKAPIBM25
print("Okapi")
w = Weighter3(ind)
#evalIR.evalModel(Okapi(w,ind))

"""

"""