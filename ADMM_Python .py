# coding=gb18030
'''
gogogo
test solomon dataset
'''

import math
import csv
import copy
import xlrd
import time
from scipy import optimize as op
import numpy as np

#the parameter need to be changed
vehicle_fleet_size =15
g_number_of_time_intervals = 960
fixed_cost=200
waiting_arc_cost=0.4
service_length=30
origin_node = 0
departure_time_beginning = 0
destination_node = 101
arrival_time_ending = 960

g_number_of_ADMM_iterations = 100

_MAX_LABEL_COST = 100000
g_node_list = []
g_agent_list = []
g_link_list = []

g_number_of_nodes = 0
g_number_of_links = 0
g_number_of_agents = 0
g_number_of_customers=0
g_number_of_vehicles=0

base_profit=200
rho = 20
constants = 100

g_ending_state_vector = [None] * g_number_of_vehicles

path_no_seq = []
path_time_seq = []
service_times = []
record_profit = []

dp_result = [_MAX_LABEL_COST, _MAX_LABEL_COST] * g_number_of_vehicles
dp_result_lowerbound = [_MAX_LABEL_COST] * g_number_of_vehicles
dp_result_upperbound = [_MAX_LABEL_COST] * g_number_of_vehicles
BestKSize = 100
global_upperbound = 9999
global_lowerbound = -9999




ADMM_local_lowerbound = [0] * g_number_of_ADMM_iterations
ADMM_local_upperbound = [0] * g_number_of_ADMM_iterations

class Node:
    def __init__(self):
        self.node_id = 0
        self.x = 0.0
        self.y = 0.0
        self.type=0
        self.outbound_node_list = []
        self.outbound_node_size=0
        self.outbound_link_list = []
        self.weight = 0.0
        self.volume = 0.0
        self.g_activity_node_beginning_time = 0
        self.g_activity_node_ending_time=0
        self.base_profit_for_searching = 0
        self.base_profit_for_lr = 0

class Link:
    def __init__(self):
        self.link_id = 0
        self.from_node_id = 0
        self.to_node_id = 0
        self.distance = 0.0
        self.spend_tm = 1.0

class Agent:
    def __init__(self):
        self.agent_id = 0
        self.from_node_id = 0
        self.to_node_id = 0
        self.departure_time_beginning = 0.0
        self.arrival_time_beginning = 0.0
        self.capacity_wei = 0
        self.capacity_vol = 0



def g_ReadInputData():
    #initialization
    global g_number_of_agents
    global g_number_of_vehicles
    global g_number_of_customers
    global g_number_of_nodes
    global g_number_of_links

    # read nodes information

    book = xlrd.open_workbook("input_node.xlsx")
    sh = book.sheet_by_index(0)
    #set the original node
    node = Node()
    node.node_id = 0
    node.type = 1
    node.g_activity_node_beginning_time = 0
    #node.g_activity_node_ending_time = 100
    g_node_list.append(node)
    g_number_of_nodes += 1


    for l in range(2,sh.nrows):  # read each lines
        try:
            node = Node()
            node.node_id = int(sh.cell_value(l,0))
            node.type=2
            node.x = float(sh.cell_value(l,2))
            node.y = float(sh.cell_value(l,3))
            node.weight = float(sh.cell_value(l,4))
            node.volume = float(sh.cell_value(l,5))
            node.g_activity_node_beginning_time = int(sh.cell_value(l,6))
            node.g_activity_node_ending_time = int(sh.cell_value(l,7))
            g_node_list.append(node)
            node.base_profit_for_searching = base_profit
            node.base_profit_for_lr = base_profit
            g_number_of_nodes += 1
            g_number_of_customers+=1
            if g_number_of_nodes % 100 == 0:
                print('reading {} nodes..' \
                      .format(g_number_of_nodes))
        except:
            print('Bad read. Check file your self')
    print('nodes_number:{}'.format(g_number_of_nodes))
    print('customers_number:{}'.format(g_number_of_customers))

    #set the destination node
    node = Node()
    node.type = 1
    node.node_id = g_number_of_nodes  #
    #node.g_activity_node_beginning_time = 0
    node.g_activity_node_ending_time = g_number_of_time_intervals
    g_node_list.append(node)
    g_number_of_nodes += 1


    with open('input_link.csv', 'r') as fl:
        linel = fl.readlines()
        for l in linel[1:]:
            l = l.strip().split(',')
            try:
                link = Link()
                link.link_id = int(l[0])
                link.from_node_id = int(l[1])
                link.to_node_id = int(l[2])
                link.distance = int(l[3])
                link.spend_tm = int(l[4])
                if (link.to_node_id ==destination_node) or\
                (link.from_node_id==origin_node)or \
                        (link.spend_tm  <= 40):
                    g_node_list[link.from_node_id].outbound_node_list.append(link.to_node_id)
                    g_node_list[link.from_node_id].outbound_node_size=len(g_node_list[link.from_node_id].outbound_node_list)
                    g_link_list.append(link)
                    g_number_of_links += 1
                    # add the outbound_link information of each node
                    g_node_list[link.from_node_id].outbound_link_list.append(link)
                if g_number_of_links % 8000 == 0:
                    print('reading {} links..' \
                          .format(g_number_of_links))
            except:
                print('Bad read. Check file your self')
        print('links_number:{}'.format(g_number_of_links))


    for i in range(vehicle_fleet_size):
        agent = Agent()
        agent.agent_id = i
        agent.from_node_id = 0
        agent.to_node_id = g_number_of_nodes
        agent.departure_time_beginning = 0
        agent.arrival_time_ending = g_number_of_time_intervals
        agent.capacity_wei = 2.0
        agent.capacity_vol = 12.0
        g_agent_list.append(agent)
        g_number_of_vehicles += 1

    print('vehicles_number:{}'.format(g_number_of_vehicles))


class CVSState:
    def __init__(self):
        self.current_node_id = 0
        self.passenger_service_state=[0]*g_number_of_nodes
        self.m_visit_sequence=[]
        self.m_visit_time_sequence=[]
        self.m_vehicle_capacity_wei = 0
        self.m_vehicle_capacity_vol = 0
        self.m_vehicle_mileage = 0
        self.LabelCost = 0     #with LR price and rho
        self.LabelCost_for_lr = 0   #with LR price
        self.PrimalLabelCost = 0    #without LR price
        self.m_final_arrival_time =0
        self.passenger_vehicle_visit_allowed_flag = [1 for i in range(g_number_of_nodes)]
        self.total_travel_cost=0
        self.total_waiting_cost = 0
        self.total_fixed_cost=0


    '''def CVSState(self):
        self.m_final_arrival_time = 0
        self.LabelCost = 0
        self.PrimalLabelCost = 0
        self.m_vehicle_capacity_wei = 0
        self.m_vehicle_capacity_vol = 0
'''

    def mycopy(self,pElement):
        self.current_node_id = copy.copy(pElement.current_node_id)
        self.passenger_service_state = []
        self.passenger_service_state = copy.copy(pElement.passenger_service_state)
        self.passenger_vehicle_visit_allowed_flag=[]
        self.passenger_vehicle_visit_allowed_flag=copy.copy(pElement.passenger_vehicle_visit_allowed_flag)
        self.m_visit_sequence = []
        self.m_visit_sequence = copy.copy(pElement.m_visit_sequence)
        self.m_visit_time_sequence = []
        self.m_visit_time_sequence = copy.copy(pElement.m_visit_time_sequence)
        self.LabelCost = copy.copy(pElement.LabelCost)
        self.LabelCost_for_lr = copy.copy(pElement.LabelCost_for_lr)
        self.PrimalLabelCost = copy.copy(pElement.PrimalLabelCost)
        self.m_vehicle_capacity_wei = copy.copy(pElement.m_vehicle_capacity_wei)
        self.m_vehicle_capacity_vol = copy.copy(pElement.m_vehicle_capacity_vol)
        self.total_travel_cost = copy.copy(pElement.total_travel_cost)
        self.total_waiting_cost = copy.copy(pElement.total_waiting_cost)
        self.total_fixed_cost = copy.copy(pElement.total_fixed_cost)


    def CalculateLabelCost(self,vehicle_id):

        #LabelCost

        # fixed_cost for each vehicle
        if from_node_id==0 and to_node_id !=g_number_of_nodes-1:
            self.LabelCost=self.LabelCost+fixed_cost
            self.LabelCost_for_lr = self.LabelCost_for_lr+fixed_cost
            self.PrimalLabelCost = self.PrimalLabelCost+fixed_cost
            self.total_fixed_cost+=fixed_cost

        # transportation_cost
        self.LabelCost = self.LabelCost-g_node_list[to_node_id].base_profit_for_searching+link_no.distance/1000.0*12.0
        self.LabelCost_for_lr = self.LabelCost_for_lr-g_node_list[to_node_id].base_profit_for_lr+link_no.distance/1000.0*12.0  #no necessary
        self.PrimalLabelCost = self.PrimalLabelCost + link_no.distance/1000.0 * 12.0
        self.total_travel_cost += link_no.distance/1000.0 * 12.0


        #waiting cost
        if from_node_id!=0 and waiting_cost_flag==1:
            self.LabelCost  = self.LabelCost +(g_node_list[to_node_id].g_activity_node_beginning_time-next_time)*waiting_arc_cost
            self.LabelCost_for_lr =  self.LabelCost_for_lr +(g_node_list[to_node_id].g_activity_node_beginning_time-next_time)*waiting_arc_cost
            self.PrimalLabelCost = self.PrimalLabelCost + (g_node_list[to_node_id].g_activity_node_beginning_time-next_time)*waiting_arc_cost
            self.total_waiting_cost+=(g_node_list[to_node_id].g_activity_node_beginning_time-next_time)*waiting_arc_cost



    '''def generate_string_key(self):
        str ='n'
        str = str + "%d"%(self.current_node_id)
       # for i in range(g_number_of_customers):
        #    if self.passenger_service_state[i]==1:
        #        str=str+ "_"+"%d"%(i)+"["+"%d"%(self.passenger_service_state[i])+"]"
        return str    '''
    def generate_string_key(self):
        str=self.current_node_id
        return str
        

class C_time_indexed_state_vector:
    def __init__(self):
        self.current_time=0
        self.m_VSStateVector=[]
        self.m_state_map=[]
    def Reset(self):
        self.current_time = 0
        self.m_VSStateVector=[]
        self.m_state_map=[]
    def m_find_state_index(self,string_key):
        if string_key in self.m_state_map:
            return self.m_state_map.index(string_key)
        else:
            return -1

    def update_state(self,new_element,ULFlag):
        string_key = new_element.generate_string_key()
        state_index = self.m_find_state_index(string_key)
        if state_index == -1:
            self.m_VSStateVector.append(new_element)
            self.m_state_map.append(string_key)
        else:
            if ULFlag ==0: #ADMM
                if new_element.LabelCost < self.m_VSStateVector[state_index].LabelCost:
                    self.m_VSStateVector[state_index] = new_element
            else:#LR(ULFlag == 1)
                if new_element.LabelCost_for_lr < self.m_VSStateVector[state_index].LabelCost_for_lr:
                    self.m_VSStateVector[state_index] = new_element




    def Sort(self,ULFlag):
        if ULFlag ==0: #ADMM
            self.m_VSStateVector=sorted(self.m_VSStateVector,key=lambda x:x.LabelCost)
        if ULFlag ==1: #LR
            self.m_VSStateVector = sorted(self.m_VSStateVector,key=lambda x:x.LabelCost_for_lr)


    def GetBestValue(self,vehicle_id):
        if len(self.m_VSStateVector) >= 1:
            return [self.m_VSStateVector[0].LabelCost_for_lr,self.m_VSStateVector[0].PrimalLabelCost,self.m_VSStateVector[0].LabelCost]


def g_optimal_time_dependenet_dynamic_programming(
        vehicle_id,
        origin_node,
        departure_time_beginning,
        destination_node,
        arrival_time_ending,
        BestKSize,
        ULFlag):

    global g_time_dependent_state_vector
    global g_ending_state_vector
    global g_vehicle_passenger_visit_flag
    global g_vehicle_passenger_visit_allowed_flag
    global link_no
    global to_node_id
    global from_node_id
    global waiting_cost_flag
    global charging_cost_flag
    global next_time

    g_time_dependent_state_vector = [[None]*(arrival_time_ending-departure_time_beginning+2)]*g_number_of_vehicles
    if arrival_time_ending > g_number_of_time_intervals or g_node_list[origin_node].outbound_node_size == 0:
        return _MAX_LABEL_COST

     #step 2: Initialization  for origin node at the preferred departure time

    for t in range(departure_time_beginning,arrival_time_ending+1):

        g_time_dependent_state_vector[vehicle_id][t] = C_time_indexed_state_vector()
        g_time_dependent_state_vector[vehicle_id][t].Reset()
        g_time_dependent_state_vector[vehicle_id][t].current_time=t


    g_ending_state_vector[vehicle_id]= C_time_indexed_state_vector()
    g_ending_state_vector[vehicle_id].Reset()
    #origin_node
    element=CVSState()
    element.current_node_id = origin_node
    g_time_dependent_state_vector[vehicle_id][departure_time_beginning].update_state(element,ULFlag)

    # step 3:dynamic programming
    #1 sort m_VSStateVector by labelCost for scan best k elements in step2
    for t in range(departure_time_beginning,arrival_time_ending):
        g_time_dependent_state_vector[vehicle_id][t].Sort(ULFlag)
        #2 scan the best k elements
        for w_index in range(min(BestKSize, len(g_time_dependent_state_vector[vehicle_id][t].m_VSStateVector))):
            pElement=g_time_dependent_state_vector[vehicle_id][t].m_VSStateVector[w_index]      #pElement is an example of  CVSState
            from_node_id=pElement.current_node_id
            # step 2.1 link from_node to to_node
            from_node=g_node_list[from_node_id]

            for i in range(from_node.outbound_node_size):
                to_node_id = from_node.outbound_node_list[i]
                to_node = g_node_list[to_node_id]
                link_no = from_node.outbound_link_list[i]
                next_time = t + link_no.spend_tm
                next_total_mileage=pElement.m_vehicle_mileage+link_no.distance
                #step 2.2 check feasibility of node type with the current element

                # to node is destination
                if to_node_id == destination_node:
                    waiting_cost_flag=0
                    charging_cost_flag = 0
                    new_element = CVSState()
                    new_element.mycopy(pElement)
                    #wait
                    new_element.m_visit_time_sequence.append(next_time)
                    new_element.m_visit_sequence.append(to_node_id)
                    new_element.m_vehicle_mileage += link_no.distance
                    #g_time_dependent_state_vector[vehicle_id][next_time].update_state(new_element)
                    new_element.m_visit_time_sequence.append(arrival_time_ending)
                    new_element.m_visit_sequence.append(to_node_id)
                    #g_time_dependent_state_vector[vehicle_id][next_time].update_state(new_element)
                    new_element.CalculateLabelCost(vehicle_id)
                    g_ending_state_vector[vehicle_id].update_state(new_element,ULFlag)
                    continue


                if to_node_id == origin_node:  # loading
                    waiting_cost_flag = 0
                    charging_cost_flag = 0

                    new_element = CVSState()
                    new_element.mycopy(pElement)
                    new_element.m_visit_time_sequence.append(next_time)
                    new_element.m_visit_sequence.append(to_node_id)
                    new_element.m_vehicle_mileage+=link_no.distance
                    new_element.m_vehicle_capacity_wei = 0
                    new_element.m_vehicle_capacity_vol = 0
                    new_element.CalculateLabelCost(vehicle_id)
                    g_time_dependent_state_vector[vehicle_id][to_node.g_activity_node_beginning_time].update_state(
                        new_element,ULFlag)
                    continue



                # to node is activity_node
                if pElement.passenger_vehicle_visit_allowed_flag[to_node_id] ==0:
                    continue
                if  pElement.passenger_vehicle_visit_allowed_flag[to_node_id] ==1:
                    if next_time > to_node.g_activity_node_ending_time:
                        continue
                    if next_time +service_length>arrival_time_ending:
                        continue
                    # feasible state transitions
                        # check capacity
                    if pElement.m_vehicle_capacity_wei > g_agent_list[vehicle_id].capacity_wei-to_node.weight:
                        continue
                    if pElement.m_vehicle_capacity_vol > g_agent_list[vehicle_id].capacity_vol-to_node.volume:
                        continue
                    # waiting
                    if next_time <to_node.g_activity_node_beginning_time:
                        waiting_cost_flag = 1
                        charging_cost_flag = 0
                        new_element = CVSState()
                        new_element.mycopy(pElement)
                        new_element.current_node_id = to_node_id
                        new_element.passenger_service_state[to_node_id] = 1
                        new_element.passenger_vehicle_visit_allowed_flag[to_node_id] = 0

                        # for arriving at activity node and begin wait
                        new_element.m_visit_time_sequence.append(next_time)
                        new_element.m_visit_sequence.append(to_node_id)
                        # for wait until activity node's depature time
                        new_element.m_vehicle_capacity_wei+=to_node.weight
                        new_element.m_vehicle_capacity_vol += to_node.volume
                        new_element.m_vehicle_mileage += link_no.distance
                        new_element.m_visit_time_sequence.append(to_node.g_activity_node_beginning_time)
                        new_element.m_visit_sequence.append(to_node_id)
                        new_element.CalculateLabelCost(vehicle_id)
                        new_element.m_visit_time_sequence.append(to_node.g_activity_node_beginning_time+service_length)
                        new_element.m_visit_sequence.append(to_node_id)
                        g_time_dependent_state_vector[vehicle_id][to_node.g_activity_node_beginning_time+service_length].update_state(new_element,ULFlag)
                        continue

                    else:
                        # donot need waiting
                        waiting_cost_flag = 0
                        charging_cost_flag = 0
                        new_element = CVSState()
                        new_element.mycopy(pElement)
                        new_element.current_node_id = to_node_id
                        new_element.passenger_service_state[to_node_id] = 1
                        new_element.m_visit_time_sequence.append(next_time)
                        new_element.m_visit_sequence.append(to_node_id)
                        new_element.m_vehicle_capacity_wei+=to_node.weight
                        new_element.m_vehicle_capacity_vol += to_node.volume
                        new_element.m_vehicle_mileage += link_no.distance
                        new_element.passenger_vehicle_visit_allowed_flag[to_node_id] = 0

                        new_element.CalculateLabelCost(vehicle_id)
                        new_element.m_visit_time_sequence.append(next_time+service_length)
                        new_element.m_visit_sequence.append(to_node_id)
                        g_time_dependent_state_vector[vehicle_id][next_time+service_length].update_state(new_element,ULFlag)
                        continue

    #print("ok")
    g_ending_state_vector[vehicle_id].Sort(ULFlag)
    print(g_ending_state_vector[vehicle_id].m_VSStateVector[0].m_visit_sequence)
    print(g_ending_state_vector[vehicle_id].m_VSStateVector[0].m_visit_time_sequence)
    return g_ending_state_vector[vehicle_id].GetBestValue(vehicle_id)



def g_Alternating_Direction_Method_of_Multipliers():
    global global_upperbound
    global global_lowerbound
    global BestKSize
    global g_ending_state_vector
    global service_state
    global base_profit
    global path_no_seq
    global path_time_seq
    global g_number_of_ADMM_iterations
    global ADMM_local_lowerbound
    global ADMM_local_upperbound
    global vehicle_fleet_size
    global service_times
    global record_profit
    global dp_result_upperbound
    global stepsize
    global repeat_served
    global un_served
    global rho
    g_ending_state_vector = [None] * g_number_of_vehicles
    repeat_served=[]
    un_served=[]

    path_no_seq=[]
    path_time_seq=[]
    service_times=[]
    service_times_lr = []
    record_profit=[]
    travel_cost=0
    waiting_cost=0
    fixed_cost=0


    dp_result=[0,0]*g_number_of_vehicles
    dp_result_lowerbound = [_MAX_LABEL_COST] * g_number_of_vehicles
    dp_result_upperbound = [_MAX_LABEL_COST] * g_number_of_vehicles
    BestKSize=100
    global_upperbound = 9999999
    global_lowerbound = -9999999


    #ADMM_local_lowerbound = [0] * g_number_of_ADMM_iterations
    #ADMM_local_upperbound = [0] * g_number_of_ADMM_iterations

    #loop for each ADMM iterations
    for i in range(g_number_of_ADMM_iterations):
        used_v = 0
        path_no_seq.append([])
        path_time_seq.append([])
        service_times.append([])
        service_times_lr.append([])
        record_profit.append([])
        repeat_served.append([])
        un_served.append([])
        for n in range(g_number_of_nodes):
            service_times[i].append(0)
            service_times_lr[i].append(0)
        print("iteration=",(i+1))
        stepsize = 1.0 /(i+1)



        #Calculate_upper_bound(i)
        # squential
        for v in range(g_number_of_vehicles):
            print("dp for vehicle %d" % (v))
            if g_ending_state_vector[v] != None:
                for n in range(1, g_number_of_nodes - 1):

                    service_times[i][n] -= g_ending_state_vector[v].m_VSStateVector[0].passenger_service_state[n]
            for n in range(1, g_number_of_nodes - 1):
                g_node_list[n].base_profit_for_searching = g_node_list[n].base_profit_for_lr + (
                            1 - 2 * service_times[i][n]) * rho

            g_optimal_time_dependenet_dynamic_programming(v, origin_node, departure_time_beginning, destination_node,
                                                          arrival_time_ending, BestKSize, 0)
            ADMM_local_upperbound[i] += g_ending_state_vector[v].m_VSStateVector[0].PrimalLabelCost
            path_no_seq[i].append(g_ending_state_vector[v].m_VSStateVector[0].m_visit_sequence)
            path_time_seq[i].append(g_ending_state_vector[v].m_VSStateVector[0].m_visit_time_sequence)

            for n in range(1, g_number_of_nodes - 1):
                service_times[i][n] += g_ending_state_vector[v].m_VSStateVector[0].passenger_service_state[n]
            if len(path_no_seq[i][v])!=2:
                used_v+=1

        #Calculate_lower_bound(i)
        g_optimal_time_dependenet_dynamic_programming(0, origin_node, departure_time_beginning, destination_node,
                                                      arrival_time_ending, BestKSize, 1)
        for v in range(g_number_of_vehicles):
            ADMM_local_lowerbound[i] += g_ending_state_vector[0].m_VSStateVector[v].LabelCost_for_lr  # v shortest paths
        for n in range(1, g_number_of_customers + 1):
            ADMM_local_lowerbound[i] = ADMM_local_lowerbound[i] + g_node_list[n].base_profit_for_lr
        g_ending_state_vector = [None] * g_number_of_vehicles


        for n in range(1, g_number_of_nodes - 1):
            if service_times[i][n]>1:
                repeat_served[i].append(n)
            if service_times[i][n]==0:
                un_served[i].append(n)

                ADMM_local_upperbound[i] = ADMM_local_upperbound[i] + 500
            record_profit[i].append(g_node_list[n].base_profit_for_lr)

        if i>15:
            if np.square(len(un_served[i])+len(repeat_served[i]))>0.25*np.square(len(un_served[i-1])+len(repeat_served[i-1])):
                rho+=10
            for n in range(1, g_number_of_nodes - 1):
                g_node_list[n].base_profit_for_lr = g_node_list[n].base_profit_for_lr + (
                        1 - service_times[i][n]) * rho
        if i<=15:
            for n in range(1, g_number_of_nodes - 1):
                g_node_list[n].base_profit_for_lr = g_node_list[n].base_profit_for_lr + (
                        1 - service_times[i][n]) * constants*stepsize

if __name__=='__main__':
    print('Reading data......')
    g_ReadInputData()
    time_start = time.time()







    g_Alternating_Direction_Method_of_Multipliers()

    time_end = time.time()
    f = open("output_path.csv", "w")
    f.write("iteration,vehicle_id,path_no_eq,path_time_sq\n")
    for i in range(g_number_of_ADMM_iterations):
        for v in range(g_number_of_vehicles):
            f.write(str(i) + ",")
            f.write(str(v) + ",")
            str1 = ""
            str2 = ""
            for s in range(len(path_no_seq[i][v])):
                str1=str1+str(path_no_seq[i][v][s])+"_"
                str2=str2+str(path_time_seq[i][v][s])+"_"
            f.write((str1)+ ","+(str2)+"\n")
    f.close()

    f = open("output_profit.csv", "w")
    f.write("iteration,")
    for n in range(1,g_number_of_customers+1):
        f.write("%d ,"%(n))
    f.write("\n" )
    for i in range(g_number_of_ADMM_iterations):
        f.write(str(i)+",")
        for n in range(g_number_of_customers):
            f.write(str(record_profit[i][n])+ ",")
        f.write("\n")


    f=open("output_gap.csv", "w")
    f.write("iteration,LB,UB,Repeated services,missed services \n")
    for i in range(g_number_of_ADMM_iterations):
        f.write(str(i)+",")
        f.write(str(ADMM_local_lowerbound[i])+ ",")
        f.write(str(ADMM_local_upperbound[i]) + ",")
        for j in repeat_served[i]:
            f.write(str(j)+";")
        f.write(",")
        for k in un_served[i]:
            f.write(str(k) + ";")
        f.write("\n")


    f.close()
    print('time cost', time_end - time_start)

