# coding=gb18030
'''
Author = Yao
SPPRC
'''

import copy
from read_input import *
import time
import cProfile

class Label:
    def __init__(self):
        self.current_node_id = -1
        self.m_visit_sequence=[1]
        self.load = 0
        self.LabelCost = 0
        self.LabelCost2 = 0
        self.PrimalLabelCost = 0
        self.forbidden_set=[]

    def generate(self,from_label,to_node,g_link_dict,next_load,ADMM_flag):
        if ADMM_flag == 1:
            to_node_price = to_node.ADMM_price
        else:
            to_node_price = to_node.dual_price
        to_node_id = to_node.node_id
        the_link = g_link_dict[(from_label.current_node_id,to_node_id)]
        self.current_node_id = to_node_id
        self.m_visit_sequence = from_label.m_visit_sequence + [to_node_id]
        self.load = next_load
        self.LabelCost =  from_label.LabelCost + the_link.distance - to_node_price - the_link.dual_price  # the dual price of links is not always necessary
        self.LabelCost2 = from_label.LabelCost + the_link.distance - to_node_price
        self.PrimalLabelCost = from_label.PrimalLabelCost + the_link.distance
        ng_set = list(set(from_label.forbidden_set).intersection(set(to_node.ng_subset)))
        ng_set.append(to_node_id)
        self.forbidden_set = ng_set

    def generate_for_heuristic(self,from_label,to_node,g_link_dict,next_load):
        to_node_id = to_node.internal_id
        self.current_node_id = to_node_id
        self.m_visit_sequence = from_label.m_visit_sequence + [to_node_id]
        self.load = next_load
        self.LabelCost =  from_label.LabelCost + link_no.distance - to_node.base_profit_for_searching - link_no.dual_price
        self.LabelCost2 = from_label.LabelCost + link_no.distance - to_node.base_profit_for_searching
        self.PrimalLabelCost = from_label.PrimalLabelCost + link_no.physical_dis
        self.forbidden_set = from_label.forbidden_set + [to_node_id]

    def _is_dominated(self,a, b):
        """returns whether a label {a} is dominated by another label {b}"""
        if a.LabelCost == b.LabelCost and a.forbidden_set==b.forbidden_set:
            label_dominated = False
        else:
            label_dominated = False
            if b.LabelCost <= a.LabelCost and set(b.forbidden_set)<= set(a.forbidden_set):
                label_dominated = True
        return label_dominated

    def is_not_dominated(self,label_bucket):
        if label_bucket==[]:
            return True
        else:
            for i in range(len(label_bucket)):
                if self._is_dominated(self, label_bucket[i]):
                    return False
        return True

    def update_labelbucket_if_the_label_dominate_others_in(self,label_bucket):
        for i in range(len(label_bucket)):
            label_dominated = self._is_dominated(label_bucket[i], self)
            if label_dominated:
                label_bucket[i] = []
        while [] in label_bucket:
            label_bucket.remove([])


def g_optimal_load_dependenet_dynamic_programming(thenetwork,veh_cap,g_node_dict,g_link_dict,T,completion_bound_flag,ADMM_flag,maintain_columns):
    origin_node = thenetwork.origin
    destination_node = thenetwork.destination
    T_current = [[float('inf') for i in range(len(thenetwork.nodes)+1)] for j in range(veh_cap + 1)]
    T_current[0][0] = 0

    label_set = [[ [] for i in range(len(thenetwork.nodes)+1)] for j in range(veh_cap + 1)]
    g_ending_state_vector = []

    original_label = Label()
    original_label.current_node_id = origin_node
    fixed_cost = 0
    original_label.LabelCost = -fixed_cost
    label_set[origin_node][0].append(original_label)

    for q in range(0, veh_cap + 1):
        for c in range(len(thenetwork.nodes)+1):
            for current_label in label_set[q][c]:
                from_node_id = current_label.current_node_id
                from_node = g_node_dict[from_node_id]
                for i in range(len(from_node.outbound_node_list)):
                    to_node_id = from_node.outbound_node_list[i]
                    if (from_node_id,to_node_id) in thenetwork.arcs:
                        if to_node_id not in current_label.forbidden_set:
                            to_node = g_node_dict[to_node_id]
                            next_load = int(current_label.load + to_node.demand)
                            if next_load <= veh_cap:
                                new_label = Label()
                                new_label.generate(current_label, to_node, g_link_dict,next_load,ADMM_flag)
                                if to_node_id == destination_node:
                                    g_ending_state_vector.append(new_label)
                                else:
                                    if ((new_label.LabelCost + T[veh_cap - next_load][to_node_id] < 0.0001) and (completion_bound_flag == 1))or (completion_bound_flag == 0):
                                        if new_label.is_not_dominated(label_set[next_load][to_node_id]):
                                            new_label.update_labelbucket_if_the_label_dominate_others_in(label_set[next_load][to_node_id])
                                            label_set[next_load][to_node_id].append(new_label)
                                            if T_current[next_load][to_node_id] > new_label.LabelCost:
                                                T_current[next_load][to_node_id] = new_label.LabelCost
    sorted_g_ending_state_vector = sorted(g_ending_state_vector, key=lambda x: x.LabelCost)

    temp_path_list = []
    temp_path_cost_list = []
    dual_price = []    #just for check
    if ADMM_flag == 1:
        return [sorted_g_ending_state_vector[0].m_visit_sequence],[sorted_g_ending_state_vector[0].LabelCost],[sorted_g_ending_state_vector[0].PrimalLabelCost],T_current
    else:
        for k in range(min(maintain_columns, len(sorted_g_ending_state_vector))):
            if sorted_g_ending_state_vector[k].LabelCost >= -0.0001:
                return temp_path_list, temp_path_cost_list, T_current
            # if g_ending_state_vector[k].m_visit_sequence[1] <= g_ending_state_vector[k].m_visit_sequence[-2]:  #single direction
            else:
                temp_path_list.append(sorted_g_ending_state_vector[k].m_visit_sequence)
                temp_path_cost_list.append(sorted_g_ending_state_vector[k].PrimalLabelCost)
                dual_price.append(sorted_g_ending_state_vector[k].LabelCost)
        return temp_path_list, temp_path_cost_list , T_current

