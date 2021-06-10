# coding=gb18030
'''
Author = Yao
'''

import copy
from read_input import *
import time
import SPPRC

def isNGRoute(cycles,g_node_dict):
    Ng_flag = True
    for c in cycles:
        Ng_flag_for_this_cycle = False
        for i in range(1,len(c)-1):
            if c[0] not in g_node_dict[c[i]].ng_set:
                Ng_flag_for_this_cycle = True
        if Ng_flag_for_this_cycle == False:
            Ng_flag = Ng_flag_for_this_cycle
            for i in c[1:-1]:
                g_node_dict[i].ng_subset.append(c[0])
    return Ng_flag

def find_cycle(path):
    cycle=[]
    re_path=[]
    for i in range(len(path)):
        re_path.append(path[len(path)-1-i])
    path_elementary = set(path)
    for i in path_elementary:
        id1=path.index(i)
        id2=re_path.index(i)
        if id1+id2 !=len(path)-1:
            cycle.append(path[id1:(len(path)-id2)])
    return cycle

def generate_completion_bound(thenetwork,veh_cap,T_current):
    T = [[float('inf') for i in range(len(thenetwork.nodes)+1)] for j in range(veh_cap + 1)]
    T[0][0]=0
    for q in range(1,veh_cap):
        for i in range(1,len(thenetwork.nodes)+1):
            T[q][i] = T_current[q][i]
            if T[q-1][i]<=T[q][i]:
                T[q][i]= T[q-1][i]
    return T

def DSSR_ng_Route_Algorithm(thenetwork,veh_cap,g_node_dict,g_link_dict,completion_bound_flag,ADMM_flag):
    # this completion_bound accerlate technique is only suitable for sysmetric graph
    for i in thenetwork.nodes:
        g_node_dict[i].ng_subset = []
    T = [[float('-inf') for i in range(len(thenetwork.nodes)+1)] for j in range(veh_cap + 1)]
    ng = 0
    ng_iteration = 1
    maintain_columns = 1
    while not ng :
        new_path, new_path_dual_cost,new_path_primal_cost,T_current =  SPPRC.g_optimal_load_dependenet_dynamic_programming(thenetwork,veh_cap,g_node_dict,g_link_dict,T,completion_bound_flag,ADMM_flag,maintain_columns)
        if new_path==[]:
            return [],[]
        ng_path = []
        ng_path_dual_cost = []
        ng_path_primal_cost = []
        cycles_in_path = []
        for p in range(len(new_path)):
            cyc = find_cycle(new_path[p][1:])
            cycles_in_path.append(cyc)
            if isNGRoute(cyc,g_node_dict):
                ng_path.append(new_path[p])
                ng_path_dual_cost.append(new_path_dual_cost[p])
                ng_path_primal_cost.append(new_path_primal_cost[p])
        if ng_path != []:
            return ng_path,ng_path_dual_cost,ng_path_primal_cost
        else:
            if completion_bound_flag == 1:
                T = generate_completion_bound(thenetwork,veh_cap,  T_current)
            ng_iteration = ng_iteration + 1
    return



