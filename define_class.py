# coding=gb18030
'''
Author = Yao
'''

class Node:
    def __init__(self):
        self.node_id = 0
        self.x = 0.0
        self.y = 0.0
        self.type = -1
        self.ng_set = []
        self.ng_subset = []
        self.outbound_node_list = []
        self.outbound_link_list = []
        self.demand = 0
        self.dual_price = 0
        self.ADMM_price = 0

class Link:
    def __init__(self):
        self.type = 0
        self.link_id = 0
        self.from_node_id = 0
        self.to_node_id = 0
        self.distance = 0.0
        self.dual_price = 0.0

class Network:
    def __init__(self,g_node_dict,g_link_dict):
        self.nodes = [i for i in g_node_dict]   # just node_id
        self.origin = 1
        self.destination = 1
        self.arcs = [i for i in g_link_dict]    # （i,j）

    def g_node_Ngset(self,ngsize,g_node_dict,g_link_dict):
        for i in g_node_dict.keys():
            temp_dict = {}
            for j in g_node_dict[i].outbound_node_list:
                temp_dict[j] = g_link_dict[(i,j)].distance
            sorted_tonode_list = list(sorted(temp_dict.items(), key=lambda x: x[1]))[:ngsize]
            g_node_dict[i].ng_set = [i[0] for i in sorted_tonode_list]
            g_node_dict[i].ng_set.append(i)
    def g_modified_nw_based_on_fixed_and_forbidden_arcs(self,fixed_arc_set,forbidden_arc_set):
        #对self.arcs进行操作，删减
        pass
