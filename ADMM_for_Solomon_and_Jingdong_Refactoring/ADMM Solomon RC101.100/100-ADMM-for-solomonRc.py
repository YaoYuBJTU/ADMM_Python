import csv
import copy
import time
import matplotlib.pyplot as plt


class Node:
    def __init__(self):
        self.node_id = 0
        self.x = 0.0
        self.y = 0.0
        self.type = 0
        self.outgoing_node_id_list = []
        self.outbound_node_size = 0
        self.outgoing_link_obj_list = []
        self.demand = 0.0
        self.service_time = 0
        self.m_activity_node_beginning_time = 0
        self.m_activity_node_ending_time = 0
        self.base_profit_for_admm = 0
        self.base_profit_for_lr = 0


class Link:
    def __init__(self):
        self.link_id = 0
        self.from_node_id = 0
        self.to_node_id = 0
        self.distance = 0.0
        self.spend_tm = 0


class Agent:
    def __init__(self):
        self.agent_id = 0
        self.from_node_id = 0
        self.to_node_id = 0
        self.departure_time_beginning = 0.0
        self.arrival_time_beginning = 0.0
        self.capacity = 0


def g_read_input_data():
    # initialization
    global g_number_of_agents
    global g_number_of_vehicles
    global g_number_of_customers
    global g_number_of_nodes
    global g_number_of_links

    # step 1: read information of NODEs

    # step 1.1: set the origin depot
    node = Node()
    node.node_id = 0
    node.type = 1
    node.x = 40.0
    node.y = 50.0
    node.m_activity_node_beginning_time = 0
    node.m_activity_node_ending_time = g_number_of_time_intervals
    g_node_list.append(node)
    g_number_of_nodes += 1

    with open("./input/input_node.csv", "r") as fp:
        print('read input_node.csv')

        reader = csv.DictReader(fp)
        for line in reader:
            node = Node()
            node.node_id = int(line["CUST NO."])
            node.type = 2
            node.x = float(line["XCOORD."])
            node.y = float(line["YCOORD."])
            node.demand = float(line["DEMAND"])
            node.m_activity_node_beginning_time = int(line["READY TIME"])
            node.m_activity_node_ending_time = int(line["DUE DATE"])
            node.service_time = int(line["SERVICE TIME"])
            node.base_profit_for_admm = base_profit
            node.base_profit_for_lr = base_profit
            g_node_list.append(node)
            g_number_of_nodes += 1
            g_number_of_customers += 1

    print(f'the number of customers is {g_number_of_customers}')

    # set the destination depot
    node = Node()
    node.type = 1
    node.node_id = g_number_of_nodes
    node.x = 40.0
    node.y = 50.0
    node.m_activity_node_beginning_time = 0
    node.m_activity_node_ending_time = g_number_of_time_intervals
    g_node_list.append(node)
    g_number_of_nodes += 1
    print(f'the number of nodes is {g_number_of_nodes}')

    # step 2: read information of LINKs
    with open("./input/input_link.csv", "r") as fp:
        print('read input_link.csv')

        reader = csv.DictReader(fp)
        for line in reader:
            link = Link()
            link.link_id = int(line["ID"])
            link.from_node_id = int(line["from_node"])
            link.to_node_id = int(line["to_node"])
            link.distance = float(line["distance"])
            link.spend_tm = int(line["spend_tm"])

            # make relationship between NODEs and LINKs
            g_node_list[link.from_node_id].outgoing_node_id_list.append(link.to_node_id)
            g_node_list[link.from_node_id].outbound_node_size = len(g_node_list[link.from_node_id].outgoing_node_id_list)
            g_node_list[link.from_node_id].outgoing_link_obj_list.append(link)

            g_link_list.append(link)
            g_number_of_links += 1
        print(f'the number of links is {g_number_of_links}')

    # step 3: read information of VEHICLEs
    for i in range(vehicle_fleet_size):
        agent = Agent()
        agent.agent_id = i
        agent.from_node_id = 0
        agent.to_node_id = g_number_of_nodes - 1
        agent.departure_time_beginning = 0
        agent.arrival_time_ending = g_number_of_time_intervals
        agent.capacity = 200
        g_agent_list.append(agent)
        g_number_of_vehicles += 1

    print(f'the number of vehicles is {g_number_of_vehicles}')


class VSState:
    def __init__(self):
        self.current_node_id = 0

        self.m_visit_sequence = []
        self.m_visit_time_sequence = []
        self.m_vehicle_used_capacity = 0

        self.passenger_service_state = [0] * g_number_of_nodes
        self.passenger_vehicle_visit_allowed_flag = [1] * g_number_of_nodes

        self.label_cost_for_admm = 0  # with LR price and rho
        self.label_cost_for_lr = 0  # with LR price
        self.primal_label_cost = 0  # without LR price

        self.total_travel_cost = 0
        self.total_waiting_cost = 0
        self.total_fixed_cost = 0

    def my_copy(self, current_element):
        self.current_node_id = current_element.current_node_id

        self.m_visit_sequence = copy.copy(current_element.m_visit_sequence)
        self.m_visit_time_sequence = copy.copy(current_element.m_visit_time_sequence)
        self.m_vehicle_used_capacity = current_element.m_vehicle_used_capacity

        self.passenger_service_state = copy.copy(current_element.passenger_service_state)
        self.passenger_vehicle_visit_allowed_flag = copy.copy(current_element.passenger_vehicle_visit_allowed_flag)

        self.label_cost_for_admm = current_element.label_cost_for_admm
        self.label_cost_for_lr = current_element.label_cost_for_lr
        self.primal_label_cost = current_element.primal_label_cost

        self.total_travel_cost = current_element.total_travel_cost
        self.total_waiting_cost = current_element.total_waiting_cost
        self.total_fixed_cost = current_element.total_fixed_cost

    def calculate_label_cost(self):
        # fixed_cost
        if from_node_id == 0 and to_node_id != g_number_of_nodes - 1:
            self.label_cost_for_admm += fixed_cost
            self.label_cost_for_lr += fixed_cost
            self.primal_label_cost += fixed_cost
            self.total_fixed_cost += fixed_cost

        # transportation_cost
        self.label_cost_for_admm = self.label_cost_for_admm - g_node_list[to_node_id].base_profit_for_admm + link_obj.distance
        self.label_cost_for_lr = self.label_cost_for_lr - g_node_list[to_node_id].base_profit_for_lr + link_obj.distance
        self.primal_label_cost = self.primal_label_cost + link_obj.distance
        self.total_travel_cost += link_obj.distance

        # waiting_cost
        if from_node_id != 0 and waiting_cost_flag == 1:
            self.label_cost_for_admm += (g_node_list[to_node_id].m_activity_node_beginning_time - next_time) * waiting_arc_cost
            self.label_cost_for_lr += (g_node_list[to_node_id].m_activity_node_beginning_time - next_time) * waiting_arc_cost
            self.primal_label_cost += (g_node_list[to_node_id].m_activity_node_beginning_time - next_time) * waiting_arc_cost
            self.total_waiting_cost += (g_node_list[to_node_id].m_activity_node_beginning_time - next_time) * waiting_arc_cost

    def generate_string_key(self):
        return self.current_node_id


class Time_Indexed_State_Vector:
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
        string_key = new_element.generate_string_key()
        state_index = self.m_find_state_index(string_key)
        if state_index == -1:
            self.m_VSStateVector.append(new_element)
            self.m_state_map.append(string_key)
        else:
            if ul_flag == 0:  # ADMM
                if new_element.label_cost_for_admm < self.m_VSStateVector[state_index].label_cost_for_admm:
                    self.m_VSStateVector[state_index] = new_element
            else:  # LR (ul_flag == 1)
                if new_element.label_cost_for_lr < self.m_VSStateVector[state_index].label_cost_for_lr:
                    self.m_VSStateVector[state_index] = new_element

    def sort(self, ul_flag):
        if ul_flag == 0:  # ADMM
            self.m_VSStateVector = sorted(self.m_VSStateVector, key=lambda x: x.label_cost_for_admm)
            self.m_state_map = [e.generate_string_key() for e in self.m_VSStateVector]

        if ul_flag == 1:  # LR
            self.m_VSStateVector = sorted(self.m_VSStateVector, key=lambda x: x.label_cost_for_lr)
            self.m_state_map = [e.generate_string_key() for e in self.m_VSStateVector]

    def get_best_value(self):
        if len(self.m_VSStateVector) >= 1:
            return [self.m_VSStateVector[0].label_cost_for_lr, self.m_VSStateVector[0].primal_label_cost, self.m_VSStateVector[0].label_cost_for_admm]


def g_time_dependent_dynamic_programming(vehicle_id,
                                         origin_node,
                                         departure_time_beginning,
                                         destination_node,
                                         arrival_time_ending,
                                         beam_width,
                                         ul_flag):
    global g_time_dependent_state_vector
    global g_ending_state_vector
    global link_obj
    global to_node_id
    global from_node_id
    global waiting_cost_flag
    global next_time

    g_time_dependent_state_vector = [None] * (arrival_time_ending - departure_time_beginning + 2)

    if arrival_time_ending > g_number_of_time_intervals or g_node_list[origin_node].outbound_node_size == 0:
        return MAX_LABEL_COST

    for t in range(departure_time_beginning, arrival_time_ending + 1):
        g_time_dependent_state_vector[t] = Time_Indexed_State_Vector()
        g_time_dependent_state_vector[t].current_time = t

    g_ending_state_vector[vehicle_id] = Time_Indexed_State_Vector()

    # origin_node
    element = VSState()
    element.current_node_id = origin_node
    g_time_dependent_state_vector[departure_time_beginning].update_state(element, ul_flag)

    for t in range(departure_time_beginning, arrival_time_ending):
        g_time_dependent_state_vector[t].sort(ul_flag)
        for w_index in range(min(beam_width, len(g_time_dependent_state_vector[t].m_VSStateVector))):
            current_element = g_time_dependent_state_vector[t].m_VSStateVector[w_index]  # current_element is an example of  VSState
            from_node_id = current_element.current_node_id
            from_node = g_node_list[from_node_id]

            for i in range(from_node.outbound_node_size):
                to_node_id = from_node.outgoing_node_id_list[i]
                to_node = g_node_list[to_node_id]
                link_obj = from_node.outgoing_link_obj_list[i]
                next_time = t + link_obj.spend_tm

                # case 1: to_node is the destination depot
                if to_node_id == destination_node:
                    waiting_cost_flag = 0
                    new_element = VSState()
                    new_element.my_copy(current_element)
                    # wait
                    new_element.m_visit_time_sequence.append(next_time)
                    new_element.m_visit_sequence.append(to_node_id)

                    new_element.m_visit_time_sequence.append(arrival_time_ending)
                    new_element.m_visit_sequence.append(to_node_id)
                    new_element.calculate_label_cost()
                    g_ending_state_vector[vehicle_id].update_state(new_element, ul_flag)
                    continue

                # case 2: to_node is the origin depot
                if to_node_id == origin_node:
                    continue

                # case 3: to_node is a customer
                if current_element.passenger_vehicle_visit_allowed_flag[to_node_id] == 0:
                    continue
                if current_element.passenger_vehicle_visit_allowed_flag[to_node_id] == 1:
                    # time window constraint
                    if next_time > to_node.m_activity_node_ending_time:
                        continue
                    if next_time + service_length > arrival_time_ending:
                        continue
                    # carrying capacity constraint
                    if current_element.m_vehicle_used_capacity > g_agent_list[vehicle_id].capacity - to_node.demand:
                        continue

                    if next_time < to_node.m_activity_node_beginning_time:  # need to wait
                        waiting_cost_flag = 1

                        new_element = VSState()
                        new_element.my_copy(current_element)
                        new_element.current_node_id = to_node_id
                        new_element.passenger_service_state[to_node_id] = 1
                        new_element.passenger_vehicle_visit_allowed_flag[to_node_id] = 0

                        new_element.m_visit_time_sequence.append(next_time)
                        new_element.m_visit_sequence.append(to_node_id)
                        new_element.m_vehicle_used_capacity += to_node.demand

                        new_element.m_visit_time_sequence.append(to_node.m_activity_node_beginning_time)
                        new_element.m_visit_sequence.append(to_node_id)

                        new_element.calculate_label_cost()

                        new_element.m_visit_time_sequence.append(to_node.m_activity_node_beginning_time + to_node.service_time)
                        new_element.m_visit_sequence.append(to_node_id)

                        g_time_dependent_state_vector[to_node.m_activity_node_beginning_time + to_node.service_time].update_state(new_element, ul_flag)
                        continue
                    else:  # do not need to wait
                        waiting_cost_flag = 0
                        new_element = VSState()
                        new_element.my_copy(current_element)
                        new_element.current_node_id = to_node_id
                        new_element.passenger_service_state[to_node_id] = 1
                        new_element.passenger_vehicle_visit_allowed_flag[to_node_id] = 0

                        new_element.m_visit_time_sequence.append(next_time)
                        new_element.m_visit_sequence.append(to_node_id)
                        new_element.m_vehicle_used_capacity += to_node.demand

                        new_element.calculate_label_cost()

                        new_element.m_visit_time_sequence.append(next_time + to_node.service_time)
                        new_element.m_visit_sequence.append(to_node_id)

                        g_time_dependent_state_vector[next_time + to_node.service_time].update_state(new_element, ul_flag)
                        continue

    g_ending_state_vector[vehicle_id].sort(ul_flag)
    return g_ending_state_vector[vehicle_id].get_best_value()


def g_alternating_direction_method_of_multipliers():
    # global variables
    global gap_threshold

    global glo_ub
    global glo_lb

    global ADMM_local_lowerbound
    global ADMM_local_upperbound

    global ADMM_global_lowerbound
    global ADMM_global_upperbound

    global beam_width
    global g_ending_state_vector
    global base_profit
    global path_node_seq
    global path_time_seq
    global g_number_of_ADMM_iterations

    global service_times
    global repeat_served
    global un_served

    global record_profit

    global rho

    g_ending_state_vector = [None] * g_number_of_vehicles

    repeat_served = []
    un_served = []

    path_node_seq = []
    path_time_seq = []
    service_times = []
    record_profit = []

    # for updating rho
    key_iter = int(g_number_of_ADMM_iterations / 3)
    primal_slack_in_last_iter = 9999  # initial: a huge number

    # loop for each ADMM iteration
    for i in range(g_number_of_ADMM_iterations):
        print(f"===  ADMM iteration number = {i}  ===")

        used_v = 0

        path_node_seq.append([])
        path_time_seq.append([])
        service_times.append([0] * g_number_of_nodes)
        record_profit.append([])
        repeat_served.append([])
        un_served.append([])

        if i != 0:
            service_times[i] = service_times[i - 1]

        # Calculate_upper_bound(i)
        for v in range(g_number_of_vehicles - 1):

            # prepare mu^v_p
            if g_ending_state_vector[v] != None:
                for n in range(1, g_number_of_customers + 1):
                    service_times[i][n] -= g_ending_state_vector[v].m_VSStateVector[0].passenger_service_state[n]

            # prepare the modified cost
            for n in range(1, g_number_of_customers + 1):
                g_node_list[n].base_profit_for_admm = g_node_list[n].base_profit_for_lr + (1 - 2 * service_times[i][n]) * rho / 2.0

            # call dynamic programming (augmented Lagrangian)
            g_time_dependent_dynamic_programming(v, origin_node, departure_time_beginning, destination_node, arrival_time_ending, beam_width, 0)

            ADMM_local_upperbound[i] += g_ending_state_vector[v].m_VSStateVector[0].primal_label_cost
            path_node_seq[i].append(g_ending_state_vector[v].m_VSStateVector[0].m_visit_sequence)
            path_time_seq[i].append(g_ending_state_vector[v].m_VSStateVector[0].m_visit_time_sequence)

            for n in range(1, g_number_of_customers + 1):
                service_times[i][n] += g_ending_state_vector[v].m_VSStateVector[0].passenger_service_state[n]

            if len(path_node_seq[i][v]) != 2:
                used_v += 1

        primal_slack = 0
        for n in range(1, g_number_of_customers + 1):
            if service_times[i][n] > 1:
                repeat_served[i].append(n)
                primal_slack += (service_times[i][n] - 1) ** 2
            if service_times[i][n] == 0:
                un_served[i].append(n)
                primal_slack += 1
                ADMM_local_upperbound[i] = ADMM_local_upperbound[i] + 90
                # ADMM_local_upperbound[i] = ADMM_local_upperbound[i] + 500
            record_profit[i].append(g_node_list[n].base_profit_for_lr)

        # Calculate_lower_bound(i)
        g_time_dependent_dynamic_programming(g_number_of_vehicles - 1, origin_node, departure_time_beginning, destination_node, arrival_time_ending, beam_width, 1)

        for vv in range(used_v):
            ADMM_local_lowerbound[i] += g_ending_state_vector[g_number_of_vehicles - 1].m_VSStateVector[vv].label_cost_for_lr
        for n in range(1, g_number_of_nodes - 1):
            ADMM_local_lowerbound[i] += g_node_list[n].base_profit_for_lr

        # adaptive updating the quadratic penalty parameter : rho
        if i >= key_iter:
            if primal_slack > 0.25 * primal_slack_in_last_iter:
                rho += 1
            if primal_slack == 0:
                rho = 1
            primal_slack_in_last_iter = primal_slack

        # sub-gradient method for updating Lagrangian multipliers
        for n in range(1, g_number_of_customers + 1):
            g_node_list[n].base_profit_for_lr = g_node_list[n].base_profit_for_lr + (1 - service_times[i][n]) * rho

        # global bound      
        if glo_ub > ADMM_local_upperbound[i]:
            glo_ub = ADMM_local_upperbound[i]
        if glo_lb < ADMM_local_lowerbound[i]:
            glo_lb = ADMM_local_lowerbound[i]
        ADMM_global_lowerbound[i] = glo_lb
        ADMM_global_upperbound[i] = glo_ub

        gap = (glo_ub - glo_lb) / glo_ub
        print(f"Gap value = {gap * 100} %")
        print(f"Global upper bound value = {glo_ub}")
        if gap < gap_threshold:
            print(f"Gap threshold satisfied, terminates! Iteration number: {i}")
            i += 1
            return i  # terminate

    return g_number_of_ADMM_iterations


def output_data(number_of_iteration):
    # step 1: output the path finding results
    f = open("./output/output_path.csv", "w")
    f.write("iteration, vehicle_id, path_node_seq, path_time_seq\n")
    for i in range(number_of_iteration):
        for v in range(g_number_of_vehicles - 1):
            f.write(str(i) + ",")
            f.write(str(v) + ",")
            str1 = ""
            str2 = ""
            for s in range(len(path_node_seq[i][v])):
                str1 = str1 + str(path_node_seq[i][v][s]) + "_"
                str2 = str2 + str(path_time_seq[i][v][s]) + "_"
            f.write(str1 + "," + str2 + "\n")
    f.close()

    # step 2: output the Lagrange multipliers
    f = open("./output/output_profit.csv", "w")
    f.write("iteration,")
    for n in range(1, g_number_of_customers + 1):
        f.write(str(n) + ",")
    f.write("\n")
    for i in range(number_of_iteration):
        f.write(str(i) + ",")
        for n in range(g_number_of_customers):
            f.write(str(record_profit[i][n]) + ",")
        f.write("\n")
    f.close()

    # step 3: output the gap information
    f = open("./output/output_gap.csv", "w")
    f.write("iteration,loc_LB,loc_UB,glo_LB,glo_UB,repeated_services,missed_services\n")
    for i in range(number_of_iteration):
        f.write(str(i) + ",")
        f.write(str(ADMM_local_lowerbound[i]) + ",")
        f.write(str(ADMM_local_upperbound[i]) + ",")
        f.write(str(ADMM_global_lowerbound[i]) + ",")
        f.write(str(ADMM_global_upperbound[i]) + ",")
        for j in repeat_served[i]:
            f.write(str(j) + "; ")
        f.write(",")
        for k in un_served[i]:
            f.write(str(k) + "; ")
        f.write("\n")
    f.close()

    # step 4: plot the global & local bound evolution curve
    iter_list = list(range(number_of_iteration))
    glob_LB_list = ADMM_global_lowerbound[:number_of_iteration]
    glob_UB_list = ADMM_global_upperbound[:number_of_iteration]
    loc_LB_list = ADMM_local_lowerbound[:number_of_iteration]
    loc_UB_list = ADMM_local_upperbound[:number_of_iteration]

    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    f = plt.figure(figsize=(20, 10))
    ax = f.add_subplot(121)
    ax.plot(iter_list, glob_LB_list, color='orange', linestyle='--')
    ax.plot(iter_list, glob_UB_list, color="red")
    ax.set_xlabel('Number of iterations', fontname="Times New Roman")
    ax.set_ylabel('Objective value', fontname="Times New Roman")
    ax.legend(labels=['Global lower bound', 'Global upper bound'], loc='best', prop={'family': 'Times New Roman'})

    ax2 = f.add_subplot(122)
    ax2.plot(iter_list, loc_LB_list, color='orange', linestyle='--')
    ax2.plot(iter_list, loc_UB_list, color="red")
    ax2.set_xlabel('Number of iterations', fontname="Times New Roman")
    ax2.set_ylabel('Objective value', fontname="Times New Roman")
    ax2.legend(labels=['Local lower bound', 'Local upper bound'], loc='best', prop={'family': 'Times New Roman'})
    f.savefig("./output/fig_gap.svg")

    # step 5: plot the path finding results
    iter_no = ADMM_global_upperbound.index(glo_ub)
    f = plt.figure(figsize=(20, 10))
    ax = f.add_subplot(121)
    for v in range(g_number_of_vehicles - 1):
        x_coord = [40]
        y_coord = [50]
        for s in range(len(path_node_seq[iter_no][v])):
            node_id = path_node_seq[iter_no][v][s]
            x_coord.append(g_node_list[node_id].x)
            y_coord.append(g_node_list[node_id].y)
        x_coord.append(40)
        y_coord.append(50)
        ax.plot(x_coord, y_coord, linewidth=1)
    ax.scatter(40, 50, marker='^')
    for n in g_node_list[1:-1]:
        ax.scatter(n.x, n.y, color="r")
    ax.set_xlabel("Longitude", fontname="Times New Roman")
    ax.set_ylabel("Latitude", fontname="Times New Roman")
    ax.set_title("Solution obtained by ADMM")

    # http://sun.aei.polsl.pl/~zjc/best-solutions-solomon.html#RC101
    opt_path = [[0, 28, 33, 85, 89, 91, 101],
                [0, 65, 52, 99, 57, 86, 74, 101],
                [0, 69, 98, 88, 53, 78, 55, 68, 101],
                [0, 27, 29, 31, 30, 34, 26, 32, 93, 101],
                [0, 92, 95, 62, 67, 71, 94, 50, 80, 101],
                [0, 64, 90, 84, 56, 66, 101],
                [0, 72, 36, 38, 41, 40, 43, 37, 35, 101],
                [0, 14, 47, 12, 73, 79, 46, 4, 60, 101],
                [0, 63, 76, 51, 22, 49, 20, 24, 101],
                [0, 59, 75, 87, 97, 58, 77, 101],
                [0, 39, 42, 44, 61, 81, 54, 96, 101],
                [0, 83, 23, 21, 19, 18, 48, 25, 101],
                [0, 82, 11, 15, 16, 9, 10, 13, 17, 101],
                [0, 5, 45, 2, 7, 6, 8, 3, 1, 70, 100, 101]]
    ax2 = f.add_subplot(122)
    for v in range(14):
        x_coord = []
        y_coord = []
        for s in range(len(opt_path[v])):
            node_id = opt_path[v][s]
            x_coord.append(g_node_list[node_id].x)
            y_coord.append(g_node_list[node_id].y)
        ax2.plot(x_coord, y_coord, linewidth=1)
    ax2.scatter(40, 50, marker='^')
    for n in g_node_list[1:-1]:
        plt.scatter(n.x, n.y, color="r")
    ax2.set_xlabel("Longitude", fontname="Times New Roman")
    ax2.set_ylabel("Latitude", fontname="Times New Roman")
    ax2.set_title("Best-known solution")
    f.savefig("./output/fig_path.svg")


if __name__ == '__main__':
    gap_threshold = 0.08  # 8 %
    MAX_LABEL_COST = 9999

    glo_ub = MAX_LABEL_COST
    glo_lb = -MAX_LABEL_COST

    g_number_of_ADMM_iterations = 150
    vehicle_fleet_size = 16
    fixed_cost = 0
    waiting_arc_cost = 0
    service_length = 10
    origin_node = 0
    departure_time_beginning = 0
    destination_node = 101
    arrival_time_ending = 240
    g_number_of_time_intervals = 240

    g_node_list = []
    g_link_list = []
    g_agent_list = []

    g_number_of_nodes = 0
    g_number_of_customers = 0
    g_number_of_links = 0
    g_number_of_agents = 0
    g_number_of_vehicles = 0

    base_profit = 0
    rho = 1

    g_ending_state_vector = [None] * g_number_of_vehicles

    path_node_seq = []
    path_time_seq = []
    service_times = []
    record_profit = []

    beam_width = 100

    ADMM_local_lowerbound = [0] * g_number_of_ADMM_iterations
    ADMM_local_upperbound = [0] * g_number_of_ADMM_iterations

    ADMM_global_lowerbound = [0] * g_number_of_ADMM_iterations
    ADMM_global_upperbound = [0] * g_number_of_ADMM_iterations

    # start
    print('Reading data......')
    g_read_input_data()

    time_start = time.time()

    number_of_iteration = g_alternating_direction_method_of_multipliers()

    print(f'processing time of admm: {time.time() - time_start:.2f} s')
    print(f"The optimal objective value is: {ADMM_global_upperbound[number_of_iteration - 1]}")

    output_data(number_of_iteration)
