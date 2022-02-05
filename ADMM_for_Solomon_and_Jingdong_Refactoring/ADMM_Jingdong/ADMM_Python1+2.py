import csv
import copy
import time


class Node:
    def __init__(self):
        self.node_id = 0
        self.x = 0.0
        self.y = 0.0
        self.type = 0
        self.outgoing_node_id_list = []
        self.outgoing_node_size = 0
        self.outgoing_link_obj_list = []
        self.weight = 0.0
        self.volume = 0.0
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
        self.capacity_wei = 0  # weight capacity
        self.capacity_vol = 0  # volume capacity


def g_read_input_data():
    global g_number_of_nodes
    global g_number_of_customers
    global g_number_of_links
    global g_number_of_vehicles

    # step 1: read NODEs information

    # step 1.1: set the origin depot
    node = Node()
    node.node_id = 0
    node.type = 1
    node.x = 116.571614
    node.y = 39.792844
    node.m_activity_node_beginning_time = 0
    node.m_activity_node_ending_time = g_number_of_time_intervals
    g_node_list.append(node)
    g_number_of_nodes += 1

    # step 1.2: set the customers
    with open('./input/input_node_100.csv', 'r') as fp:
        print('read input_node_100.csv')

        reader = csv.DictReader(fp)
        for line in reader:
            node = Node()
            node.node_id = int(line["ID"])
            node.type = 2
            node.x = float(line["lng"])
            node.y = float(line["lat"])
            node.weight = float(line["pack_total_weight"])
            node.volume = float(line["pack_total_volume"])
            node.m_activity_node_beginning_time = int(line["first_receive_tm"])
            node.m_activity_node_ending_time = int(line["last_receive_tm"])
            g_node_list.append(node)
            node.base_profit_for_admm = base_profit
            node.base_profit_for_lr = base_profit
            g_number_of_nodes += 1
            g_number_of_customers += 1
    print('customers_number:{}'.format(g_number_of_customers))

    # step 1.3: set the destination node
    node = Node()
    node.type = 1
    node.node_id = g_number_of_nodes
    node.x = 116.571614
    node.y = 39.792844
    node.m_activity_node_beginning_time = 0
    node.m_activity_node_ending_time = g_number_of_time_intervals
    g_node_list.append(node)
    g_number_of_nodes += 1

    print('nodes_number:{}'.format(g_number_of_nodes))

    # step 2: read LINKs information
    with open('./input/input_link_100.csv', 'r') as fp:
        print('read input_link_100.csv')

        reader = csv.DictReader(fp)
        for line in reader:
            link = Link()
            link.link_id = int(line["ID"])
            link.from_node_id = int(line["from_node"])
            link.to_node_id = int(line["to_node"])
            link.distance = int(line["distance"])
            link.spend_tm = int(line["spend_tm"])
            if (link.to_node_id == destination_node) or (link.from_node_id == origin_node) or (link.spend_tm <= 50):
                g_node_list[link.from_node_id].outgoing_node_id_list.append(link.to_node_id)
                g_node_list[link.from_node_id].outgoing_node_size = len(g_node_list[link.from_node_id].outgoing_node_id_list)
                g_node_list[link.from_node_id].outgoing_link_obj_list.append(link)
                g_link_list.append(link)
                g_number_of_links += 1

        print('links_number:{}'.format(g_number_of_links))

    # step 3: set VEHICLEs information
    for i in range(vehicle_fleet_size):
        agent = Agent()
        agent.agent_id = i
        agent.from_node_id = 0
        agent.to_node_id = g_number_of_nodes - 1
        agent.departure_time_beginning = 0
        agent.arrival_time_ending = g_number_of_time_intervals
        agent.capacity_wei = 2.0
        agent.capacity_vol = 12.0
        g_agent_list.append(agent)
        g_number_of_vehicles += 1

    print('vehicles_number:{}'.format(g_number_of_vehicles))


class VSState:
    def __init__(self):
        self.current_node_id = 0
        self.m_visit_node_sequence = []
        self.m_visit_time_sequence = []
        self.m_vehicle_capacity_wei = 0
        self.m_vehicle_capacity_vol = 0

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
        self.m_visit_node_sequence = copy.copy(current_element.m_visit_node_sequence)
        self.m_visit_time_sequence = copy.copy(current_element.m_visit_time_sequence)
        self.m_vehicle_capacity_wei = current_element.m_vehicle_capacity_wei
        self.m_vehicle_capacity_vol = current_element.m_vehicle_capacity_vol

        self.passenger_service_state = copy.copy(current_element.passenger_service_state)
        self.passenger_vehicle_visit_allowed_flag = copy.copy(current_element.passenger_vehicle_visit_allowed_flag)

        self.label_cost_for_admm = current_element.label_cost_for_admm
        self.label_cost_for_lr = current_element.label_cost_for_lr
        self.primal_label_cost = current_element.primal_label_cost

        self.total_travel_cost = current_element.total_travel_cost
        self.total_waiting_cost = current_element.total_waiting_cost
        self.total_fixed_cost = current_element.total_fixed_cost

    def calculate_label_cost(self):

        # fixed_cost for each vehicle
        if from_node_id == 0 and to_node_id != g_number_of_nodes - 1:
            self.label_cost_for_admm += + fixed_cost
            self.label_cost_for_lr += fixed_cost
            self.primal_label_cost += fixed_cost
            self.total_fixed_cost += fixed_cost

        # transportation_cost
        self.label_cost_for_admm = self.label_cost_for_admm - g_node_list[to_node_id].base_profit_for_admm + link_obj.distance / 1000.0 * 12.0
        self.label_cost_for_lr = self.label_cost_for_lr - g_node_list[to_node_id].base_profit_for_lr + link_obj.distance / 1000.0 * 12.0  # no necessary
        self.primal_label_cost = self.primal_label_cost + link_obj.distance / 1000.0 * 12.0
        self.total_travel_cost += link_obj.distance / 1000.0 * 12.0

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
            else:  # LR(ul_flag == 1)
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
    global g_vehicle_passenger_visit_flag
    global g_vehicle_passenger_visit_allowed_flag
    global link_obj
    global to_node_id
    global from_node_id
    global waiting_cost_flag
    global next_time

    g_time_dependent_state_vector = [None] * (arrival_time_ending - departure_time_beginning + 2)
    if arrival_time_ending > g_number_of_time_intervals or g_node_list[origin_node].outgoing_node_size == 0:
        return _MAX_LABEL_COST

    # step 2: Initialization  for origin node at the preferred departure time

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
            current_element = g_time_dependent_state_vector[t].m_VSStateVector[w_index]
            from_node_id = current_element.current_node_id
            from_node = g_node_list[from_node_id]

            for i in range(from_node.outgoing_node_size):
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
                    new_element.m_visit_node_sequence.append(to_node_id)
                    new_element.m_visit_time_sequence.append(arrival_time_ending)
                    new_element.m_visit_node_sequence.append(to_node_id)
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
                    # capacity constraint
                    if current_element.m_vehicle_capacity_wei > g_agent_list[vehicle_id].capacity_wei - to_node.weight:
                        continue
                    if current_element.m_vehicle_capacity_vol > g_agent_list[vehicle_id].capacity_vol - to_node.volume:
                        continue
                    if next_time < to_node.m_activity_node_beginning_time:  # need to wait
                        waiting_cost_flag = 1
                        new_element = VSState()
                        new_element.my_copy(current_element)
                        new_element.current_node_id = to_node_id
                        new_element.passenger_service_state[to_node_id] = 1
                        new_element.passenger_vehicle_visit_allowed_flag[to_node_id] = 0

                        new_element.m_visit_time_sequence.append(next_time)
                        new_element.m_visit_node_sequence.append(to_node_id)
                        new_element.m_vehicle_capacity_wei += to_node.weight
                        new_element.m_vehicle_capacity_vol += to_node.volume
                        new_element.m_visit_time_sequence.append(to_node.m_activity_node_beginning_time)
                        new_element.m_visit_node_sequence.append(to_node_id)

                        new_element.calculate_label_cost()
                        new_element.m_visit_time_sequence.append(to_node.m_activity_node_beginning_time + service_length)
                        new_element.m_visit_node_sequence.append(to_node_id)
                        g_time_dependent_state_vector[to_node.m_activity_node_beginning_time + service_length].update_state(new_element, ul_flag)
                        continue
                    else:  # do not need waiting
                        waiting_cost_flag = 0
                        new_element = VSState()
                        new_element.my_copy(current_element)
                        new_element.current_node_id = to_node_id
                        new_element.passenger_service_state[to_node_id] = 1
                        new_element.passenger_vehicle_visit_allowed_flag[to_node_id] = 0

                        new_element.m_visit_time_sequence.append(next_time)
                        new_element.m_visit_node_sequence.append(to_node_id)
                        new_element.m_vehicle_capacity_wei += to_node.weight
                        new_element.m_vehicle_capacity_vol += to_node.volume

                        new_element.calculate_label_cost()
                        new_element.m_visit_time_sequence.append(next_time + service_length)
                        new_element.m_visit_node_sequence.append(to_node_id)
                        g_time_dependent_state_vector[next_time + service_length].update_state(new_element, ul_flag)
                        continue

    g_ending_state_vector[vehicle_id].sort(ul_flag)
    return g_ending_state_vector[vehicle_id].get_best_value()


def g_alternating_direction_method_of_multipliers():
    global gap_threshold
    global beam_width
    global g_ending_state_vector
    global base_profit
    global path_node_seq
    global path_time_seq
    global g_number_of_ADMM_iterations
    global global_upperbound
    global global_lowerbound
    global ADMM_local_lowerbound
    global ADMM_local_upperbound
    global ADMM_global_lowerbound
    global ADMM_global_upperbound
    global service_times
    global record_profit
    global repeat_served
    global un_served
    global rho

    g_ending_state_vector = [None] * g_number_of_vehicles
    repeat_served = []
    un_served = []

    path_node_seq = []
    path_time_seq = []
    service_times = []
    record_profit = []

    beam_width = 100

    global_upperbound = 9999999
    global_lowerbound = -9999999

    # for updating rho
    # key_iter = int(g_number_of_ADMM_iterations / 3)
    # key_iter = 50
    # primal_slack_in_last_iter = 9999  # initial: a huge number

    # loop for each ADMM iteration
    for i in range(g_number_of_ADMM_iterations):
        used_v = 0
        path_node_seq.append([])
        path_time_seq.append([])
        service_times.append([0] * g_number_of_nodes)
        record_profit.append([])
        repeat_served.append([])
        un_served.append([])
        if i != 0:
            service_times[i] = service_times[i - 1]
        print(f"ADMM Iteration no = {i}")

        # Calculate_upper_bound(i)
        for v in range(g_number_of_vehicles - 1):
            if g_ending_state_vector[v] != None:
                for n in range(1, g_number_of_nodes - 1):
                    service_times[i][n] -= g_ending_state_vector[v].m_VSStateVector[0].passenger_service_state[n]
            for n in range(1, g_number_of_nodes - 1):
                g_node_list[n].base_profit_for_admm = g_node_list[n].base_profit_for_lr + (1 - 2 * service_times[i][n]) * rho / 2.0

            g_time_dependent_dynamic_programming(v, origin_node, departure_time_beginning, destination_node, arrival_time_ending, beam_width, 0)

            ADMM_local_upperbound[i] += g_ending_state_vector[v].m_VSStateVector[0].primal_label_cost
            path_node_seq[i].append(g_ending_state_vector[v].m_VSStateVector[0].m_visit_node_sequence)
            path_time_seq[i].append(g_ending_state_vector[v].m_VSStateVector[0].m_visit_time_sequence)

            for n in range(1, g_number_of_nodes - 1):
                service_times[i][n] += g_ending_state_vector[v].m_VSStateVector[0].passenger_service_state[n]
            if len(path_node_seq[i][v]) != 2:
                used_v += 1

        # primal_slack = 0
        for n in range(1, g_number_of_nodes - 1):
            if service_times[i][n] > 1:
                repeat_served[i].append(n)
                # primal_slack += (service_times[i][n] - 1) ** 2
            if service_times[i][n] == 0:
                un_served[i].append(n)
                # primal_slack += 1
                ADMM_local_upperbound[i] = ADMM_local_upperbound[i] + 400
            record_profit[i].append(g_node_list[n].base_profit_for_lr)

        # Calculate_lower_bound(i)
        g_time_dependent_dynamic_programming(g_number_of_vehicles - 1, origin_node, departure_time_beginning, destination_node, arrival_time_ending, beam_width, 1)

        for vv in range(g_number_of_vehicles - 1):
            ADMM_local_lowerbound[i] = ADMM_local_lowerbound[i] + min(g_ending_state_vector[g_number_of_vehicles - 1].m_VSStateVector[vv].label_cost_for_lr, 0)  # v shortest paths
        for n in range(1, g_number_of_nodes - 1):
            ADMM_local_lowerbound[i] = ADMM_local_lowerbound[i] + g_node_list[n].base_profit_for_lr

        # self-adaptive updating the quadratic penalty parameter : rho
        # the following implementation is modified based on Dr. Yu Yao
        if i >= 50:
            if (len(un_served[i]) + len(repeat_served[i])) ** 2 > 0.25 * (len(un_served[i - 1]) + len(repeat_served[i - 1])) ** 2:
                rho += 2
            if (len(un_served[i]) + len(repeat_served[i])) ** 2 == 0:
                rho = 1

        ## update rho, implemented by Chongnan Li, which is more consistent with the Part B paper
        # if i >= key_iter:
        #     if primal_slack > 0.25 * primal_slack_in_last_iter:
        #         rho += 2
        #     if primal_slack == 0:
        #         rho = 1
        #     primal_slack_in_last_iter = primal_slack

        for n in range(1, g_number_of_nodes - 1):
            g_node_list[n].base_profit_for_lr = g_node_list[n].base_profit_for_lr + (1 - service_times[i][n]) * rho

        if global_upperbound > ADMM_local_upperbound[i]:
            global_upperbound = ADMM_local_upperbound[i]
        if global_lowerbound < ADMM_local_lowerbound[i]:
            global_lowerbound = ADMM_local_lowerbound[i]
        ADMM_global_upperbound[i] = global_upperbound
        ADMM_global_lowerbound[i] = global_lowerbound

        rel_gap = (global_upperbound - global_lowerbound) / global_upperbound
        if rel_gap < gap_threshold and len(repeat_served[i]) == 0 and len(un_served[i]) == 0:
            i += 1
            print("gap threshold satisfied, terminate ADMM")
            print(f"gap = {rel_gap * 100} %")
            print(f"obj value = {global_upperbound}")
            return i

    print("max iter num reached, terminate ADMM")
    print(f"gap = {rel_gap * 100} %")
    print(f"obj value = {global_upperbound}")
    return g_number_of_ADMM_iterations


if __name__ == '__main__':
    gap_threshold = 0.075  # 7.5%

    vehicle_fleet_size = 16
    g_number_of_time_intervals = 960
    fixed_cost = 200
    waiting_arc_cost = 0.4
    service_length = 30

    origin_node = 0
    departure_time_beginning = 0
    destination_node = 100
    arrival_time_ending = 960

    g_number_of_ADMM_iterations = 200

    _MAX_LABEL_COST = 100000

    g_node_list = []
    g_link_list = []
    g_agent_list = []

    g_number_of_nodes = 0
    g_number_of_customers = 0
    g_number_of_links = 0
    g_number_of_vehicles = 0

    base_profit = 150
    rho = 1

    g_ending_state_vector = [None] * g_number_of_vehicles

    path_node_seq = []
    path_time_seq = []
    service_times = []
    record_profit = []

    beam_width = 100

    global_upperbound = 999999
    global_lowerbound = -999999
    ADMM_global_lowerbound = [0] * g_number_of_ADMM_iterations
    ADMM_global_upperbound = [0] * g_number_of_ADMM_iterations
    ADMM_local_lowerbound = [0] * g_number_of_ADMM_iterations
    ADMM_local_upperbound = [0] * g_number_of_ADMM_iterations

    print('Reading data......')
    g_read_input_data()
    time_start = time.time()

    num_iter = g_alternating_direction_method_of_multipliers()

    time_end = time.time()
    f = open("./output/output_path.csv", "w")
    f.write("iteration,vehicle_id,path_node_seq,path_time_seq\n")
    for i in range(num_iter):
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

    f = open("./output/output_profit.csv", "w")
    f.write("iteration,")
    for n in range(1, g_number_of_customers + 1):
        f.write(str(n) + ",")
    f.write("\n")
    for i in range(num_iter):
        f.write(str(i) + ",")
        for n in range(g_number_of_customers):
            f.write(str(record_profit[i][n]) + ",")
        f.write("\n")
    f.close()

    f = open("./output/output_gap.csv", "w")
    f.write("iteration,loc_LB,loc_UB,glob_LB,glob_UB,repeated_services,missed_services \n")
    for i in range(num_iter):
        f.write(str(i) + ",")
        f.write(str(ADMM_local_lowerbound[i]) + ",")
        f.write(str(ADMM_local_upperbound[i]) + ",")
        f.write(str(ADMM_global_lowerbound[i]) + ",")
        f.write(str(ADMM_global_upperbound[i]) + ",")
        for j in repeat_served[i]:
            f.write(str(j) + ";")
        f.write(",")
        for k in un_served[i]:
            f.write(str(k) + ";")
        f.write("\n")
    f.close()

    print('time cost', time_end - time_start)
