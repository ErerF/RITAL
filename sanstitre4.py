# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:42:19 2020

@author: arnau
"""

from TextRep import *

txtRep = TextRepresenter()
stemmer = PorterStemmer()
doc = "The new home has been saled on top forecasts"
res = stemmer.getTextRepresentation(doc)
print(res)
txtRep.getTextRepresentation()