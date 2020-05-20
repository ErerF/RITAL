# -*- coding: utf-8 -*-
"""
Created on Wed May 20 11:55:46 2020

@author: arnau
"""
import Parsing
import TME3


p = Parser()
p.parsing("../cacmShort-good.txt")
print(p.collection)

ind = IndexerSimple()
ind.indexation(p.collection)
