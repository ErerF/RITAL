# -*- coding: utf-8 -*-


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
"""for d in queries:
    print(d.I)
    print(len(d.listRelevantDocs))
"""

class EvalMesure():
    
    def evalQuery(self,liste,query):
        raise NotImplementedError()
        
class Precision(EvalMesure):
    
    def __init__(self,rang):
        self.rang = rang
        
    def evalQuery(self,liste,query):
        if len(liste) > self.rang:
            liste = liste[:self.rang-1]
        precision = 0
        for doc in liste:
            if doc in query.listRelevantDocs:
                precision += 1
        return precision/self.rang

class Rappel(EvalMesure):
    
    def __init__(self,rang):
        self.rang = rang
        
    def evalQuery(self,liste,query):
        if len(query.listRelevantDocs) == 0:
            return 1
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
        if P+R == 0:
            return 0
        else:
            return 2*P*R/(P+R)
    
class PrecisionMoyenne(EvalMesure):
    """
    Somme des produits Precision à rang k fois le document au rang k est pertinent ou non
    Divisé par le nombre de docs pertinents à la query
    """
    def evalQuery(self,liste,query):
        numberRelevantDocs = max(len(query.listRelevantDocs),1)
        avgP = 0
        
        for i in range(1,len(liste)):
            evalPrecision = Precision(i)
            pak = evalPrecision.evalQuery(liste,query)
            
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
        """
        ON sélectionne un modele puis pour chaque query on effectue les mesures
        A la fin on fait la moyenne et l'ecart type de chaque mesure
        """
        IRModel = model
        
        rang = 30
        
        listPrecision = []
        listRappel = []
        listFmesure = []
        listPrecisionmoyenne = []
        listReciprocalRank = []
        listNDCG = []
        
        Mprecision = Precision(rang)
        Mrappel = Rappel(rang)
        Mfmesure = Fmesure(rang)
        Mprecisionmoyenne = PrecisionMoyenne()
        Mreciprocalrank = ReciprocalRank()
        Mndcg = NDCG()
        
        for q in queries:
            IRModel.getScores(q.W)
            listeMeilleursDocs = [int(x[0]) for x in IRModel.getRanking()[:rang]]
            listPrecision.append(Mprecision.evalQuery(listeMeilleursDocs,q))
            listRappel.append(Mrappel.evalQuery(listeMeilleursDocs,q))
            listFmesure.append(Mfmesure.evalQuery(listeMeilleursDocs,q))
            listPrecisionmoyenne.append(Mprecisionmoyenne.evalQuery(listeMeilleursDocs,q))
            listReciprocalRank.append(Mreciprocalrank.evalQuery(listeMeilleursDocs,q))
            listNDCG.append(Mndcg.evalQuery(listeMeilleursDocs,q))
            
            
        listPrecision = np.array(listPrecision)
        listRappel = np.array(listRappel)
        listFmesure = np.array(listFmesure)
        listPrecisionmoyenne = np.array(listPrecisionmoyenne)
        listReciprocalRank = np.array(listReciprocalRank)
        listNDCG = np.array(listNDCG)
        
        print("Precision")
        print("Moyenne :")
        print(np.mean(listPrecision))
        print("Ecart-type :")
        print(np.std(listPrecision))
        
        print("Rappel")
        print("Moyenne :")
        print(np.mean(listRappel ))
        print("Ecart-type :")
        print(np.std(listRappel ))
        
        print("Fmesure")
        print("Moyenne :")
        print(np.mean(listFmesure))
        print("Ecart-type :")
        print(np.std(listFmesure))
        
        print("Precision Moyenne")
        print("Moyenne :")
        print(np.mean(listPrecisionmoyenne))
        print("Ecart-type :")
        print(np.std(listPrecisionmoyenne))
        
        print("ReciprocalRank")
        print("Moyenne :")
        print(np.mean(listReciprocalRank))
        print("Ecart-type :")
        print(np.std(listReciprocalRank))
        
        print("NDCG")
        print("Moyenne :")
        print(np.mean(listNDCG))
        print("Ecart-type :")
        print(np.std(listNDCG))
        
        return

    


p = Parser()
p.parsing("cisi/cisi.txt")

ind = IndexerSimple()
ind.indexation(p.collection)

parser = QueryParser()
queries = parser.parseQueries("cisi/cisi.qry","cisi/cisi.rel")



evalIR = EvalIRModel(ind,queries)
"""
#IRMODEL VECTORIEL COSINUS
print()
print("Vectoriel Cosinus")
print()
w = Weighter1(ind)
evalIR.evalModel(Vectoriel(w,ind,True))
"""
"""
Vectoriel Cosinus

Precision
Moyenne :
0.09077380952380952
Ecart-type :
0.12362014763524813
Rappel
Moyenne :
0.41468855586236825
Ecart-type :
0.42353611251458867
Fmesure
Moyenne :
0.07396424808819921
Ecart-type :
0.0889841561901804
Precision Moyenne
Moyenne :
0.013412960690101934
Ecart-type :
0.03136535780102013
ReciprocalRank
Moyenne :
0.27974642781089043
Ecart-type :
0.36810680660489536
NDCG
Moyenne :
0.0908136149990298
Ecart-type :
0.11541572809130357
"""
"""
#IRMODEL VECTORIEL SCALAIRE
print() 
print("Vectoriel scalaire")
print()
w = Weighter1(ind)
evalIR.evalModel(Vectoriel(w,ind,False))
"""
"""
Vectoriel scalaire

Precision
Moyenne :
0.07976190476190477
Ecart-type :
0.11013674246506626
Rappel
Moyenne :
0.4009705436058253
Ecart-type :
0.4298107039698095
Fmesure
Moyenne :
0.06400415743423274
Ecart-type :
0.07626234642010946
Precision Moyenne
Moyenne :
0.009535185244139247
Ecart-type :
0.022195854498189012
ReciprocalRank
Moyenne :
0.24047796461288526
Ecart-type :
0.31151503578979417
NDCG
Moyenne :
0.08484188809333289
Ecart-type :
0.12998759573191657
"""

#IRMODEL ModeleLangue 
print()
print("ModeleLangue")
w = Weighter2(ind)
evalIR.evalModel(ModeleLangue(w,ind))

"""

"""
"""
#IRMODEL OKAPIBM25
print("Okapi")
w = Weighter3(ind)
evalIR.evalModel(Okapi(w,ind))
"""
"""
Okapi
Precision
Moyenne :
0.11726190476190478
Ecart-type :
0.1405757818883279
Rappel
Moyenne :
0.4337562616117352
Ecart-type :
0.4134936315008862
Fmesure
Moyenne :
0.09498098765753042
Ecart-type :
0.10611341396727181
Precision Moyenne
Moyenne :
0.02259500560807356
Ecart-type :
0.04265458466447942
ReciprocalRank
Moyenne :
0.317616465380907
Ecart-type :
0.3742075105840503
NDCG
Moyenne :
0.1226032280247092
Ecart-type :
0.14755188892602036
"""

    
p = Parser()
p.parsing("cacm/cacm.txt")

ind = IndexerSimple()
ind.indexation(p.collection)

parser = QueryParser()
queries = parser.parseQueries("cacm/cacm.qry","cacm/cacm.rel")



evalIR = EvalIRModel(ind,queries)
"""
#IRMODEL VECTORIEL COSINUS
print()
print("Vectoriel Cosinus")
print()
w = Weighter1(ind)
evalIR.evalModel(Vectoriel(w,ind,True))
"""
"""
Vectoriel Cosinus

Precision
Moyenne :
0.0875
Ecart-type :
0.0932626220233308
Rappel
Moyenne :
0.39976736679504343
Ecart-type :
0.3465700644490704
Fmesure
Moyenne :
0.10782276594848939
Ecart-type :
0.10395229763452302
Precision Moyenne
Moyenne :
0.023944586133460906
Ecart-type :
0.038865683319838674
ReciprocalRank
Moyenne :
0.3688502606676949
Ecart-type :
0.41008131562603345
NDCG
Moyenne :
0.17589017269639068
Ecart-type :
0.19173351659523313
"""
"""
#IRMODEL VECTORIEL SCALAIRE
print() 
print("Vectoriel scalaire")
print()
w = Weighter1(ind)
evalIR.evalModel(Vectoriel(w,ind,False))
"""
"""
Vectoriel scalaire

Precision
Moyenne :
0.03697916666666667
Ecart-type :
0.05499911220874388
Rappel
Moyenne :
0.2618297368227356
Ecart-type :
0.3678037589878269
Fmesure
Moyenne :
0.044086145599013665
Ecart-type :
0.05856696412588218
Precision Moyenne
Moyenne :
0.005566246053160331
Ecart-type :
0.015316482521211585
ReciprocalRank
Moyenne :
0.182443748662179
Ecart-type :
0.2864644483634904
NDCG
Moyenne :
0.06832060886110497
Ecart-type :
0.10415413948442406
"""

#IRMODEL ModeleLangue 
print()
print("ModeleLangue")
w = Weighter2(ind)
evalIR.evalModel(ModeleLangue(w,ind))

"""

"""
"""
#IRMODEL OKAPIBM25
print("Okapi")
w = Weighter3(ind)
evalIR.evalModel(Okapi(w,ind))
"""
"""
Okapi
Precision
Moyenne :
0.10364583333333333
Ecart-type :
0.10946796644262345
Rappel
Moyenne :
0.46197121904560806
Ecart-type :
0.3588515771240857
Fmesure
Moyenne :
0.12821256228538064
Ecart-type :
0.11899251841547707
Precision Moyenne
Moyenne :
0.03921620240147769
Ecart-type :
0.06378720275957384
ReciprocalRank
Moyenne :
0.40303075396825394
Ecart-type :
0.39326062104830045
NDCG
Moyenne :
0.23146745801322316
Ecart-type :
0.2232571679144899
"""