# coding=gb18030
__author__ = 'Yao'


import time
import paraVRP
from read_input import *

from RMP import *
from CG import *
import Branch_and_Bound
import cProfile
import ADMM_based_framework

time_limit = 900
if __name__ == "__main__":
    print("Read input and Generate network...")
    instanceVRP = paraVRP.ParaVRP()
    instanceVRP.initParams()
    start_time = time.time()
    print("ADMM Starts")
    ADMMTree = ADMM_based_framework.ADMM(200)
    ADMMTree.conductADMM(instanceVRP)
    end_time = time.time()
    print("Total CPU time = ",end_time-start_time)




