import csv
import time
import copy
import pandas as pd
import matplotlib.pyplot as plt


class Node:
    """
    class name: Node
    represents physical nodes in the vehicle routing system
    including the origin node (depot, type=1), the destination node (depot, type=1), and the customers (type=2)
    """

    def __init__(self):
        """
        attributes of the Node object
        """
        self.node_id = 0
        self.x = 0.0
        self.y = 0.0
        self.type = 0
        self.outgoing_node_id_list = []
        self.outgoing_node_size = 0
        self.outgoing_link_obj_list = []
        self.demand = 0.0
        self.activity_beginning_time = 0
        self.activity_ending_time = 0
        self.service_time = 0
        self.base_profit_for_admm = 0
        self.base_profit_for_lr = 0


class Link:
    """
    class name: Link
    represents the physical links in the vehicle routing system
    the nodes incident to a link can be any physical node, including customers, origin depot, and destination depot
    """

    def __init__(self):
        """
        attributes of the Link object
        """
        self.link_id = 0
        self.from_node_id = 0
        self.to_node_id = 0
        # Q: how to compute the distance of a link?
        # A: ((customer1['x_coord'] - customer2['x_coord']) ** 2 + (customer1['y_coord'] - customer2['y_coord']) ** 2) ** 0.5
        self.distance = 0.0
        self.spend_tm = 1.0


class Agent:
    """
    class name: Agent
    An Agent object represents a vehicle, which is a decision maker to carry out the logistics transportation missions
    """

    def __init__(self):
        """
        attributes of the Agent object
        """
        self.agent_id = 0
        self.from_node_id = 0
        self.to_node_id = 0
        self.departure_time_beginning = 0
        self.arrival_time_ending = 0
        self.capacity = 0


class VSState:
    """
    class Name: VSState
    class for Vehicle Scheduling State
    member functions:
        __init__(): generate an VSState object
        my_copy(): given a VSState object, we can load the information from that VSState object through this function
        calculate_label_cost: calculate the label cost of the VSState object, it is used for label updating, objective function computation, and bound value finding
        generate_string_key: generate the string key of the VSState object
    """

    def __init__(self):
        self.current_node_id = 0
        self.m_visit_node_sequence = []
        self.m_visit_time_sequence = []
        self.m_used_vehicle_capacity = 0

        self.passenger_service_state = [0] * g_number_of_nodes
        self.passenger_vehicle_visit_allowed_flag = [1] * g_number_of_nodes

        self.label_cost_for_admm = 0
        self.label_cost_for_lr = 0
        self.primal_label_cost = 0

        self.total_travel_cost = 0
        self.total_waiting_cost = 0
        self.total_fixed_cost = 0

    def my_copy(self, current_element):
        self.current_node_id = current_element.current_node_id
        self.m_visit_node_sequence = copy.copy(current_element.m_visit_node_sequence)
        self.m_visit_time_sequence = copy.copy(current_element.m_visit_time_sequence)
        self.m_used_vehicle_capacity = current_element.m_used_vehicle_capacity

        self.passenger_service_state = copy.copy(current_element.passenger_service_state)
        self.passenger_vehicle_visit_allowed_flag = copy.copy(current_element.passenger_vehicle_visit_allowed_flag)

        self.label_cost_for_admm = current_element.label_cost_for_admm
        self.label_cost_for_lr = current_element.label_cost_for_lr
        self.primal_label_cost = current_element.primal_label_cost  # primal label cost is USED to compute the objective function value to the PRIMAL problem (namely, the upper bound value)

        self.total_travel_cost = current_element.total_travel_cost
        self.total_waiting_cost = current_element.total_waiting_cost
        self.total_fixed_cost = current_element.total_fixed_cost

    def calculate_label_cost(self):
        # fixed cost
        if from_node_id == 0 and to_node_id != g_number_of_nodes - 1:
            self.label_cost_for_admm += fixed_cost
            self.label_cost_for_lr += fixed_cost
            self.primal_label_cost += fixed_cost
            self.total_fixed_cost += fixed_cost

        # transportation cost
        self.label_cost_for_admm = self.label_cost_for_admm - g_node_list[to_node_id].base_profit_for_admm + link_obj.distance
        self.label_cost_for_lr = self.label_cost_for_lr - g_node_list[to_node_id].base_profit_for_lr + link_obj.distance
        self.primal_label_cost = self.primal_label_cost + link_obj.distance
        self.total_travel_cost = self.total_travel_cost + link_obj.distance

        # waiting cost
        if from_node_id != 0 and waiting_cost_flag == 1:
            self.label_cost_for_admm += (g_node_list[to_node_id].activity_beginning_time - next_time) * waiting_arc_cost
            self.label_cost_for_lr += (g_node_list[to_node_id].activity_beginning_time - next_time) * waiting_arc_cost
            self.primal_label_cost += (g_node_list[to_node_id].activity_beginning_time - next_time) * waiting_arc_cost
            self.total_waiting_cost += (g_node_list[to_node_id].activity_beginning_time - next_time) * waiting_arc_cost

    def generate_string_key(self):
        return self.current_node_id


class TimeIndexedStateVector:
    """
    class Name: TimeIndexedStateVector
    vector recording states at different time instants
    VSState objects will be stored in the member variables of this class
    """

    def __init__(self):
        self.current_time = 0
        self.m_VSStateVector = []
        self.m_state_map = []

    def m_find_state_index(self, string_key):
        if string_key in self.m_state_map:
            return self.m_state_map.index(string_key)
        else:
            return -1

    def update_state(self, new_element, ul_flag):

        string_key = new_element.generate_string_key()  # obtain the "string_key" of VSState object "new_element"
        state_index = self.m_find_state_index(string_key)  # try to find the location of the "string_key" within "m_state_map" object

        if state_index == -1:
            self.m_VSStateVector.append(new_element)  # just add the VSState object "new_element" into "m_VSStateVector"
            self.m_state_map.append(string_key)  # add the "string_key" into "m_state_map"
        else:  # (state_index != -1)
            if ul_flag == 0:  # ADMM
                if new_element.label_cost_for_admm < self.m_VSStateVector[state_index].label_cost_for_admm:
                    self.m_VSStateVector[state_index] = new_element
            else:  # LR
                if new_element.label_cost_for_lr < self.m_VSStateVector[state_index].label_cost_for_lr:
                    self.m_VSStateVector[state_index] = new_element

    def sort(self, ul_flag):
        if ul_flag == 0:  # ADMM
            self.m_VSStateVector = sorted(self.m_VSStateVector, key=lambda x: x.label_cost_for_admm)
            self.m_state_map = [e.generate_string_key() for e in self.m_VSStateVector]
        else:  # LR
            self.m_VSStateVector = sorted(self.m_VSStateVector, key=lambda x: x.label_cost_for_lr)
            self.m_state_map = [e.generate_string_key() for e in self.m_VSStateVector]

    def get_best_value(self):
        if len(self.m_VSStateVector) >= 1:
            return [self.m_VSStateVector[0].label_cost_for_admm, self.m_VSStateVector[0].label_cost_for_lr, self.m_VSStateVector[0].primal_label_cost]


def read_input_data():
    # global variables
    global g_number_of_nodes
    global g_number_of_customers
    global g_number_of_links
    global g_number_of_agents
    global g_number_of_time_intervals

    # step 1: read information of NODEs

    # step 1.1: establish origin depot
    node = Node()
    node.node_id = 0
    node.type = 1
    node.x = 40.0
    node.y = 50.0
    node.activity_beginning_time = 0
    node.activity_ending_time = g_number_of_time_intervals
    g_node_list.append(node)
    g_number_of_nodes += 1

    # step 1.2: establish customers
    with open(r"./input/input_node.csv", "r") as fp:
        print('Read input_node.csv')

        reader = csv.DictReader(fp)
        for line in reader:
            node = Node()

            node.node_id = int(line['NO.'])
            node.type = 2
            node.x = float(line['XCOORD.'])
            node.y = float(line['YCOORD.'])
            node.demand = float(line['DEMAND'])
            node.activity_beginning_time = int(line['READYTIME'])
            node.activity_ending_time = int(line['DUEDATE'])
            node.service_time = int(line['SERVICETIME'])

            g_node_list.append(node)
            g_number_of_nodes += 1
            g_number_of_customers += 1

        print(f'The number of customers is {g_number_of_customers}')

    # step 1.3: establish destination depot
    node = Node()
    node.node_id = g_number_of_nodes
    node.type = 1
    node.x = 40.0
    node.y = 50.0
    node.activity_beginning_time = 0
    node.activity_ending_time = g_number_of_time_intervals
    g_node_list.append(node)
    g_number_of_nodes += 1

    print(f'the number of nodes is {g_number_of_nodes}')

    # step 2: read information of LINKs
    with open(r"./input/input_link.csv", "r") as fp:
        print('Read input_link.csv')

        reader = csv.DictReader(fp)
        for line in reader:
            link = Link()

            link.link_id = int(line['ID'])
            link.from_node_id = int(line['from_node'])
            link.to_node_id = int(line['to_node'])
            link.distance = float(line['distance'])
            link.spend_tm = int(line['spend_tm'])

            # establish the correlation with nodes and links
            g_node_list[link.from_node_id].outgoing_node_id_list.append(link.to_node_id)  # add the ID of the tail Node object into the "outbound_node_list" of the head Node object
            g_node_list[link.from_node_id].outgoing_node_size = len(g_node_list[link.from_node_id].outgoing_node_id_list)  # update the "outbound_node_size" of the head Node object
            g_node_list[link.from_node_id].outgoing_link_obj_list.append(link)  # add the Link object into the "outbound_link_list" of the head node of the Link instance

            g_link_list.append(link)
            g_number_of_links += 1

        print(f'The number of links is {g_number_of_links}')

    # step 3: read information of AGENTs
    for i in range(g_number_of_agents):
        agent = Agent()

        agent.agent_id = i
        agent.from_node_id = 0
        agent.to_node_id = g_number_of_nodes - 1
        agent.departure_time_beginning = 0
        agent.arrival_time_ending = g_number_of_time_intervals
        agent.capacity = 200

        g_agent_list.append(agent)

    print(f'The number of agents is {g_number_of_agents}')


def g_time_dependent_dynamic_programming(vehicle_id, origin_node, departure_time_beginning, destination_node, arrival_time_ending, beam_width, ul_flag):
    # :param ULFlag: 0 or 1, controls whether the dynamic programming is for ADMM algorithm or the pure Lagrangian relaxation
    #                 0: ADMM (Upper Bound) ; 1: LR (Lower Bound)

    # global variables
    global g_ending_state_vector
    global waiting_cost_flag  # 0: no need to wait; 1: need to wait
    global link_obj
    global from_node_id
    global to_node_id
    global next_time

    g_time_dependent_state_vector = [None] * (arrival_time_ending - departure_time_beginning + 1)

    if arrival_time_ending > g_number_of_time_intervals or g_node_list[origin_node].outgoing_node_size == 0:
        return MAX_LABEL_COST

    for t in range(departure_time_beginning, arrival_time_ending + 1):
        g_time_dependent_state_vector[t] = TimeIndexedStateVector()
        g_time_dependent_state_vector[t].current_time = t

    g_ending_state_vector[vehicle_id] = TimeIndexedStateVector()

    # origin node
    element = VSState()
    element.current_node_id = origin_node
    g_time_dependent_state_vector[departure_time_beginning].update_state(element, ul_flag)

    # start dynamic programming
    for t in range(departure_time_beginning, arrival_time_ending):
        g_time_dependent_state_vector[t].sort(ul_flag)

        for w in range(min(beam_width, len(g_time_dependent_state_vector[t].m_VSStateVector))):
            current_element = g_time_dependent_state_vector[t].m_VSStateVector[w]
            from_node_id = current_element.current_node_id
            from_node = g_node_list[from_node_id]

            for i in range(from_node.outgoing_node_size):
                to_node_id = from_node.outgoing_node_id_list[i]
                to_node = g_node_list[to_node_id]
                link_obj = from_node.outgoing_link_obj_list[i]
                next_time = t + link_obj.spend_tm

                # case i: to_node is the destination depot
                if to_node_id == destination_node:
                    waiting_cost_flag = 0  # no need to wait

                    new_element = VSState()
                    new_element.my_copy(current_element)

                    new_element.m_visit_node_sequence.append(to_node_id)
                    new_element.m_visit_time_sequence.append(next_time)

                    new_element.m_visit_node_sequence.append(to_node_id)
                    new_element.m_visit_time_sequence.append(arrival_time_ending)

                    new_element.calculate_label_cost()

                    g_ending_state_vector[vehicle_id].update_state(new_element, ul_flag)
                    continue

                # case ii: to_node is the origin depot
                if to_node_id == origin_node:
                    continue

                # case iii: to_node is a customer
                if current_element.passenger_vehicle_visit_allowed_flag[to_node_id] == 0:
                    # has been visited, no allow
                    continue
                else:  # current_element.passenger_vehicle_visit_allowed_flag[to_node_id] == 1
                    # time window constraint
                    if next_time > to_node.activity_ending_time:
                        continue
                    if next_time + to_node.service_time > arrival_time_ending:
                        continue

                    # capacity constraint
                    if current_element.m_used_vehicle_capacity + to_node.demand > g_agent_list[vehicle_id].capacity:
                        continue

                    # check whether it is needed to wait
                    if next_time < to_node.activity_beginning_time:  # need to wait
                        waiting_cost_flag = 1

                        new_element = VSState()
                        new_element.my_copy(current_element)

                        new_element.current_node_id = to_node_id
                        new_element.passenger_service_state[to_node_id] = 1  # change the entry of "passenger_service_state" to note that the "to_node" has been visited
                        new_element.passenger_vehicle_visit_allowed_flag[to_node_id] = 0  # change the corresponding flag of "passenger_vehicle_visit_allowed_flag" to note that the vehicle "vehicle_id" can not visit this "to_node" again

                        new_element.m_visit_node_sequence.append(to_node_id)
                        new_element.m_visit_time_sequence.append(next_time)

                        new_element.m_used_vehicle_capacity += to_node.demand

                        new_element.m_visit_node_sequence.append(to_node_id)
                        new_element.m_visit_time_sequence.append(to_node.activity_beginning_time)

                        new_element.calculate_label_cost()

                        new_element.m_visit_node_sequence.append(to_node_id)
                        new_element.m_visit_time_sequence.append(to_node.activity_beginning_time + to_node.service_time)

                        g_time_dependent_state_vector[to_node.activity_beginning_time + to_node.service_time].update_state(new_element, ul_flag)
                        continue
                    else:  # do NOT need to wait
                        waiting_cost_flag = 0

                        new_element = VSState()
                        new_element.my_copy(current_element)

                        new_element.current_node_id = to_node_id
                        new_element.passenger_service_state[to_node_id] = 1  # change the entry of "passenger_service_state" to note that the "to_node" has been visited
                        new_element.passenger_vehicle_visit_allowed_flag[to_node_id] = 0  # change the corresponding flag of "passenger_vehicle_visit_allowed_flag" to note that the vehicle "vehicle_id" can not visit this "to_node" again

                        new_element.m_visit_node_sequence.append(to_node_id)
                        new_element.m_visit_time_sequence.append(next_time)

                        new_element.m_used_vehicle_capacity += to_node.demand

                        new_element.calculate_label_cost()

                        new_element.m_visit_node_sequence.append(to_node_id)
                        new_element.m_visit_time_sequence.append(next_time + to_node.service_time)

                        g_time_dependent_state_vector[next_time + to_node.service_time].update_state(new_element, ul_flag)
                        continue

    g_ending_state_vector[vehicle_id].sort(ul_flag)

    return g_ending_state_vector[vehicle_id].get_best_value()


def g_alternating_direction_method_of_multipliers():
    # global variables
    global admm_local_lower_bound
    global admm_local_upper_bound

    global admm_global_lower_bound
    global admm_global_upper_bound

    global glo_lb
    global glo_ub

    global beam_width

    global path_node_seq
    global path_time_seq

    global g_number_of_admm_iterations

    global g_number_of_agents
    global g_number_of_nodes

    global service_times
    global repeat_served
    global un_served

    global record_profit

    global rho

    path_node_seq = []
    path_time_seq = []

    service_times = []
    repeat_served = []
    un_served = []

    record_profit = []

    global_upper_bound = MAX_LABEL_COST
    global_lower_bound = -MAX_LABEL_COST

    for i in range(g_number_of_admm_iterations):
        print(f"=== Iteration number for the ADMM: {i} ===")

        number_of_used_vehicles = 0

        path_node_seq.append([])
        path_time_seq.append([])

        service_times.append([0] * g_number_of_nodes)

        repeat_served.append([])
        un_served.append([])

        record_profit.append([0] * g_number_of_nodes)

        if i > 0:
            service_times[i] = service_times[i - 1]

        for v in range(g_number_of_agents - 1):
            print(f"Dynamic programming for vehicle: {v}")

            # prepare mu^v_p
            if g_ending_state_vector[v] != None:
                for n in range(1, g_number_of_customers + 1):
                    service_times[i][n] -= g_ending_state_vector[v].m_VSStateVector[0].passenger_service_state[n]

            for n in range(1, g_number_of_customers + 1):
                g_node_list[n].base_profit_for_admm = g_node_list[n].base_profit_for_lr + (1 - 2 * service_times[i][n]) * rho / 2.0

            vehicle = g_agent_list[v]
            g_time_dependent_dynamic_programming(v, vehicle.from_node_id, vehicle.departure_time_beginning, vehicle.to_node_id, vehicle.arrival_time_ending, beam_width, 0)

            admm_local_upper_bound[i] += g_ending_state_vector[v].m_VSStateVector[0].primal_label_cost
            path_node_seq[i].append(g_ending_state_vector[v].m_VSStateVector[0].m_visit_node_sequence)
            path_time_seq[i].append(g_ending_state_vector[v].m_VSStateVector[0].m_visit_time_sequence)

            for n in range(1, g_number_of_customers + 1):
                service_times[i][n] += g_ending_state_vector[v].m_VSStateVector[0].passenger_service_state[n]

            if len(path_node_seq[i][v]) != 2:
                number_of_used_vehicles += 1

        for n in range(1, g_number_of_customers + 1):
            if service_times[i][n] > 1:
                repeat_served[i].append(n)
            if service_times[i][n] == 0:
                un_served[i].append(n)
                admm_local_upper_bound[i] += 50
                # number_of_used_vehicles += 1
            record_profit[i].append(g_node_list[n].base_profit_for_lr)

        print(f"Number of used vehicles: {number_of_used_vehicles}")

        vehicle = g_agent_list[-1]
        g_time_dependent_dynamic_programming(vehicle.agent_id, vehicle.from_node_id, vehicle.departure_time_beginning, vehicle.to_node_id, vehicle.arrival_time_ending, beam_width, 1)
        admm_local_lower_bound[i] += g_number_of_agents * g_ending_state_vector[g_number_of_agents - 1].m_VSStateVector[0].label_cost_for_lr

        for n in range(1, g_number_of_customers + 1):
            admm_local_lower_bound[i] += g_node_list[n].base_profit_for_lr
            g_node_list[n].base_profit_for_lr += (1 - service_times[i][n]) * rho

        if glo_ub > admm_local_upper_bound[i]:
            glo_ub = admm_local_upper_bound[i]
        admm_global_upper_bound[i] = glo_ub

        if glo_lb < admm_local_lower_bound[i]:
            glo_lb = admm_local_lower_bound[i]
        admm_global_lower_bound[i] = glo_lb


def output_data():
    # output path finding result
    f = open("./output/output_path.csv", "w")
    f.write("iteration,vehicle_id,path_node_seq,path_time_seq,\n")
    for i in range(g_number_of_admm_iterations):
        for v in range(g_number_of_agents - 1):
            f.write(str(i) + ",")  # iteration number of admm: "i"
            f.write(str(v) + ",")  # ID of the vehicle: "v"
            str1 = ""  # string which records the sequence of nodes in the path
            str2 = ""  # string which records the sequence of time instants in the path
            for s in range(len(path_node_seq[i][v])):
                str1 = str1 + str(path_node_seq[i][v][s]) + "_"
                str2 = str2 + str(path_time_seq[i][v][s]) + "_"
            f.write(str1 + "," + str2 + ",\n")
    f.close()

    # output the Lagrangian multipliers
    f = open("./output/output_profit.csv", "w")
    f.write("iteration,")
    for n in range(1, g_number_of_customers + 1):
        f.write(str(n) + ",")
    f.write("\n")
    for i in range(g_number_of_admm_iterations):
        f.write(str(i) + ",")
        for n in range(g_number_of_customers):
            f.write(str(record_profit[i][n]) + ",")
        f.write("\n")
    f.close()

    # output the gap information
    f = open("./output/output_gap.csv", "w")
    f.write("iteration,local_lower_bound,local_upper_bound,global_lower_bound,global_upper_bound,repeated_services,missed_services,\n")
    for i in range(g_number_of_admm_iterations):
        f.write(str(i) + ",")  # write the current iteration number
        f.write(str(admm_local_lower_bound[i]) + ",")  # write the local lower bound value for the current iteration
        f.write(str(admm_local_upper_bound[i]) + ",")  # write the local upper bound value for the current iteration
        f.write(str(admm_global_lower_bound[i]) + ",")  # write the global upper bound value for the current iteration
        f.write(str(admm_global_upper_bound[i]) + ",")  # write the global upper bound value for the current iteration
        for j in repeat_served[i]:
            f.write(str(j) + "; ")
        f.write(",")
        for k in un_served[i]:
            f.write(str(k) + "; ")
        f.write(",\n")
    f.close()

    # plot
    gap_df = pd.read_csv("./output/output_gap.csv")
    iter_list = list(gap_df['iteration'])
    glo_LB_list = list(gap_df['global_lower_bound'])
    glo_UB_list = list(gap_df['global_upper_bound'])
    loc_LB_list = list(gap_df['local_lower_bound'])
    loc_UB_list = list(gap_df['local_upper_bound'])

    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300

    plt.figure()
    plt.plot(iter_list, glo_LB_list, color='orange', linestyle='--')
    plt.plot(iter_list, glo_UB_list, color="red")
    plt.xlabel('Number of iterations', fontname="Times New Roman")
    plt.ylabel('Objective value', fontname="Times New Roman")
    plt.legend(labels=['Global lower bound', 'Global upper bound'], loc='best', prop={'family': 'Times New Roman'})
    plt.savefig("./output/fig_global_gap.svg")
    plt.show()

    plt.figure()
    plt.plot(iter_list, loc_LB_list, color='orange', linestyle='--')
    plt.plot(iter_list, loc_UB_list, color="red")
    plt.xlabel('Number of iterations', fontname="Times New Roman")
    plt.ylabel('Objective value', fontname="Times New Roman")
    plt.legend(labels=['Local lower bound', 'Local upper bound'], loc='best', prop={'family': 'Times New Roman'})
    plt.savefig("./output/fig_local_gap.svg")
    plt.show()

    # plot the path finding result (spatial)
    plt.figure()
    for v in range(g_number_of_agents - 1):
        x_coord = [40]
        y_coord = [50]
        for s in range(len(path_node_seq[g_number_of_admm_iterations - 1][v])):
            # obtain the Node object
            node_ID = path_node_seq[-1][v][s]
            x_coord.append(g_node_list[node_ID].x)
            y_coord.append(g_node_list[node_ID].y)
        x_coord.append(40)
        y_coord.append(50)
        plt.plot(x_coord, y_coord, linewidth=0.5)

    # plot the planar illustration
    plt.scatter(40, 50, marker='^')
    x_coord = []
    y_coord = []
    for n in g_node_list[1:-1]:
        x_coord.append(n.x)
        y_coord.append(n.y)

    plt.xlabel("Longitude", fontname="Times New Roman")
    plt.ylabel("Latitude", fontname="Times New Roman")
    plt.scatter(x_coord, y_coord)

    plt.savefig("./output/fig_path.svg")
    plt.show()


if __name__ == "__main__":
    fixed_cost = 0
    waiting_arc_cost = 0

    g_number_of_nodes = 0
    g_number_of_customers = 0
    g_number_of_links = 0
    g_number_of_agents = 11  # this value is 11, the best-known solution utilizes 10 vehicles
    # 10 agents are used to compute the upper bound (Admm) , 1 agent is used to compute the lower bound (LR)
    g_number_of_time_intervals = 1236
    g_number_of_admm_iterations = 16

    MAX_LABEL_COST = 99999

    beam_width = 100

    rho = 1

    g_node_list = []
    g_link_list = []
    g_agent_list = []

    admm_local_lower_bound = [0] * g_number_of_admm_iterations  # lower bound value of each iteration in the ADMM algorithm
    admm_local_upper_bound = [0] * g_number_of_admm_iterations  # upper bound value of each iteration in the ADMM algorithm

    admm_global_lower_bound = [0] * g_number_of_admm_iterations  # lower bound value of each iteration in the ADMM algorithm
    admm_global_upper_bound = [0] * g_number_of_admm_iterations  # upper bound value of each iteration in the ADMM algorithm

    glo_lb = -99999
    glo_ub = 99999

    g_ending_state_vector = [None] * g_number_of_agents

    print("Reading data......")
    read_input_data()
    time_start = time.time()

    g_alternating_direction_method_of_multipliers()

    print(f'Processing time of ADMM: {time.time() - time_start: .2f} s')

    output_data()
