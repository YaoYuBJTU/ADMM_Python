# coding=gb18030
import xlrd
import numpy
from read_input import *
from datetime import datetime

class ParaVRP:
    def __init__(self):
        self.input_file = ".\\P-VRP\\P-n16-k8.vrp"
        self.g_node_dict = {}
        self.g_link_dict = {}
        self.number_of_vehicles = 8
        self.rho = 1
        self.origin_node = -1
        self.destination_node = -1

    def initParams(self):
        self.g_node_dict, self.g_link_dict = g_ReadInputData(self)

    # with open(self.output_result_file, "w")as f:
    #     now = datetime.now()
    #     writer = csv.writer(f, delimiter=",", lineterminator='\n')
    #     writer.writerow(['start time = {: %X}\n'.format(now)])
    #     writer.writerow(["obj", "cols", "cuts", "x_solution", "y_solution", "time"])
    # with open(self.output_result_file2, "w")as f:
    #     now = datetime.now()
    #     writer = csv.writer(f, delimiter=",", lineterminator='\n')
    #     writer.writerow(['start time = {: %X}\n'.format(now)])
    #     writer.writerow(["Iteration", "ub", "lb", "Time"])
