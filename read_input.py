# coding=gb18030
'''
Author = Yao
'''

from define_class import *
import numpy
import re
# read input 改成能读取通用数据集的，测试更方便
def g_ReadInputData(paraVRP):
    print("Read input nodes ...")
    f = open(paraVRP.input_file)
    data_list = f.readlines()
    #定位到node坐标和node需求位置
    start_line_of_location = list(filter(lambda x: data_list[x] == 'NODE_COORD_SECTION\n', list(range(len(data_list)))))[0]
    start_line_of_demand = list(filter(lambda x: data_list[x] == 'DEMAND_SECTION\n', list(range(len(data_list)))))[0]
    start_line_of_depot =  list(filter(lambda x: data_list[x] == 'DEPOT_SECTION\n', list(range(len(data_list)))))[0]
    for i in range(start_line_of_location):
        line = data_list[i]
        if ('NAME' in line):
            paraVRP.file_name = re.split(": |\n",line)[1]
        elif ('DIMENSION' in line):
            paraVRP.num_customers = int(re.split(": |\n",line)[1])
        elif ('CAPACITY' in line):
            paraVRP.veh_cap = int(re.split(": |\n",line)[1])
        else:
            pass
    for i in range(start_line_of_location+1,start_line_of_demand):
        line = data_list[i]
        node_id,node_x,node_y = int(line.split()[0]),int(line.split()[1]),int(line.split()[2])
        node = Node()
        node.node_id = node_id
        node.x = node_x
        node.y = node_y
        paraVRP.g_node_dict[node_id] = node
    for i in range(start_line_of_demand+1,start_line_of_depot):
        line = data_list[i]
        node_id, node_demand = int(line.split()[0]), int(line.split()[1])
        paraVRP.g_node_dict[node_id].demand = node_demand
    paraVRP.origin_node = int(data_list[start_line_of_depot+1])
    paraVRP.destination_node = int(data_list[start_line_of_depot+1])

    # generate links
    print("Generate links ...")
    for i in paraVRP.g_node_dict.keys():
        for j in paraVRP.g_node_dict.keys():
            if i != j:
                link = Link()
                link.from_node_id = i
                link.to_node_id = j
                link.distance = int(numpy.sqrt(numpy.square(paraVRP.g_node_dict[i].x - paraVRP.g_node_dict[j].x) + numpy.square(
                    paraVRP.g_node_dict[i].y - paraVRP.g_node_dict[j].y)))
                link.spend_tm = int(link.distance)
                paraVRP.g_link_dict[(i, j)] = link
                paraVRP.g_node_dict[i].outbound_node_list.append(j)
    link = Link()
    link.from_node_id = paraVRP.origin_node
    link.to_node_id = paraVRP.destination_node
    link.distance = 0
    link.spend_tm = 0
    paraVRP.g_link_dict[(paraVRP.origin_node, paraVRP.destination_node)] = link
    paraVRP.g_node_dict[paraVRP.origin_node].outbound_node_list.append(paraVRP.destination_node)
    return  paraVRP.g_node_dict,paraVRP.g_link_dict



