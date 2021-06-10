# coding=gb18030
__author__ = 'Yao'
import DSSR_ng_route
from define_class import *
import numpy as np
import copy
class ADMM_ITER():
    def  __init__(self):
        self.number_of_iteration = 0
        self.lower_bound_solution = []
        self.upper_bound_solution = []
        self.feasible_solution = []
        self.lower_bound_value = 0
        self.upper_bound_value = float('inf')
        self.repeated_nodes = []
        self.unserved_nodes = []
        self.served_time_dict = {}



class ADMM():
    def __init__(self, Iteration_limit):
        self.priority_queue = []
        self.incumbent = float('inf')
        self.incumbent_node_id = -1
        self.path_dict = {}
        self.number_of_paths = 0
        self.iteration_limit = Iteration_limit
        self.time_limit_reached = False
        self.obj_value = float('inf')
        self.Lagrangian_price = {}
        self.lower_bound = 0
        self.upper_bound  = float('inf')
        self.admm_itr_list = []

    def check_feasibility(self,served_time_dict):
        repeated_nodes = []
        unserved_nodes = []
        for i in served_time_dict:
            if served_time_dict[i] > 1:
                repeated_nodes.append(i)
            if served_time_dict[i] == 0:
                unserved_nodes.append(i)
        return repeated_nodes,unserved_nodes

    def g_served_time_for_each_node(self):
        served_time_dict = {}
        return served_time_dict


    def g_feasible_solution_based_on_upper_bound(self,current_solution):

        pass

    def update_Lagrangian_price(self,instanceVRP,served_time_dict):
        # repeated_nodes,unserved_nodes = self.check_feasibility()
        # served_time_dict = self.g_served_time_for_each_node()
        for i in instanceVRP.g_node_dict:
            if i != 1:
                instanceVRP.g_node_dict[i].dual_price += (1-served_time_dict[i]) * instanceVRP.rho
                instanceVRP.g_node_dict[i].ADMM_price = instanceVRP.g_node_dict[i].dual_price


    def update_rho(self,itr,admm_itr,instanceVRP,served_time_dict):
        admm_itr.repeated_nodes, admm_itr.unserved_nodes = self.check_feasibility(served_time_dict)
        try:
            previous_admm_itr = self.admm_itr_list[itr-1]
            if np.square(len(admm_itr.repeated_nodes)+len(admm_itr.unserved_nodes)) > 0.25*np.square(len(previous_admm_itr.repeated_nodes)+len(previous_admm_itr.unserved_nodes)):
                instanceVRP.rho=instanceVRP.rho * 2
            if np.square(len(admm_itr.repeated_nodes)+len(admm_itr.unserved_nodes))==0:
                instanceVRP.rho = instanceVRP.rho / 2
        except:
            pass
    def update_served_time_dict_and_ADMM_price_by_removing_one_block(self,v,instanceVRP,served_time_dict,g_node_dict,path_list_for_last_iteration):
        if path_list_for_last_iteration != []:
            path_to_remove = path_list_for_last_iteration[v][0]
            for i in path_to_remove[1:-1]:
                served_time_dict[i] -= 1
        for i in served_time_dict:
                g_node_dict[i].ADMM_price = g_node_dict[i].dual_price+( 1-2*served_time_dict[i]) *instanceVRP.rho / 2

    def update_served_time_dict_and_ADMM_price_by_adding_one_block(self,served_time_dict,path):
        for i in path[1:-1]:
            served_time_dict[i] += 1

    def update_ADMM_price_during_one_iteration(self,v,instanceVRP,served_time_dict,g_node_dict,path_list_for_last_iteration,path):
        path_to_add = path
        for i in path_to_add[1:-1]:
            served_time_dict[i] += 1
        if path_list_for_last_iteration != []:
            path_to_remove = path_list_for_last_iteration[v][0]
            for i in served_time_dict:
                if i in path_to_remove:
                    served_time_dict[i] -= 1
        for i in served_time_dict:
                g_node_dict[i].ADMM_price += ( 1-2*served_time_dict[i]) *instanceVRP.rho / 2
        return served_time_dict
    def calculate_upper_bound(self,upper_bound_solution):
        upper_bound = 0
        for p in upper_bound_solution:
            upper_bound += p[2]
        return upper_bound
        # self.g_feasible_solution_based_on_upper_bound(upper_bound_solution)

    def calculate_lower_bound(self,lower_bound_solution,g_node_dict):
        path_cost = lower_bound_solution[0][1]
        local_lower_bound = path_cost * number_of_used_vehicle + [i.dual_price for i in g_node_dict]
        return local_lower_bound

    def conductADMM(self,instanceVRP):
        path = [[]]
        thenetwork = Network(instanceVRP.g_node_dict,instanceVRP.g_link_dict)
        thenetwork.g_node_Ngset(len(instanceVRP.g_node_dict),instanceVRP.g_node_dict,instanceVRP.g_link_dict)
        served_time_dict = dict.fromkeys(range(2,len(instanceVRP.g_node_dict)+1), 0)
        print(served_time_dict)
        for itr in range(self.iteration_limit):
            print("iteration=", (itr + 1))
            admm_itr = ADMM_ITER()
            try:
                previous_admm_itr = self.admm_itr_list[-1]
            except:
                previous_admm_itr = ADMM_ITER()
                # served_time_dict = {}
                # for i in instanceVRP.g_node_dict:
                #     if i !=thenetwork.origin and i != thenetwork.destination:
                #         served_time_dict[i] = 0
            admm_itr.number_of_iteration = itr
            lb_path,lb_path_cost,lb_path_primal_cost = DSSR_ng_route.DSSR_ng_Route_Algorithm(thenetwork,instanceVRP.veh_cap,instanceVRP.g_node_dict,instanceVRP.g_link_dict,completion_bound_flag=1,ADMM_flag=1) #for lower bound
            admm_itr.lower_bound_solution.append([lb_path[0],lb_path_cost[0]])
            for v in range(instanceVRP.number_of_vehicles):
                self.update_served_time_dict_and_ADMM_price_by_removing_one_block(v,instanceVRP,served_time_dict,instanceVRP.g_node_dict,previous_admm_itr.upper_bound_solution)
                path, path_dual_cost,path_primal_cost = DSSR_ng_route.DSSR_ng_Route_Algorithm(thenetwork, instanceVRP.veh_cap, instanceVRP.g_node_dict,instanceVRP.g_link_dict, completion_bound_flag=1,ADMM_flag=1)
                self.update_served_time_dict_and_ADMM_price_by_adding_one_block(served_time_dict,path[0])

                admm_itr.upper_bound_solution.append([path[0], path_dual_cost[0],path_primal_cost[0]])
            admm_itr.upper_bound_value = self.calculate_upper_bound(admm_itr.upper_bound_solution)
            # admm_itr.lower_bound_value = self.calculate_lower_bound(admm_itr.lower_bound_solution,instanceVRP.g_node_dict)
            self.update_Lagrangian_price(instanceVRP,served_time_dict)
            # self.update_rho(itr,admm_itr,instanceVRP,served_time_dict)
            admm_itr.served_time_dict = copy.copy(served_time_dict)
            self.admm_itr_list.append(admm_itr)
        print(admm_itr.upper_bound_solution)





