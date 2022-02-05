# Solomon C101.100

C101.100 is a dataset of Vehicle Routing Problems with Time Windows (VRPTW), created by Solomon.

The basic information of C101.100 is:

Number of depots: 1

Number of customers: 100

Service time of each customer: 90

Carrying capacity of vehicles: 200

The time window of the depot is: [0, 1236]

Customers are clusterly distributed in the area.

# ADMM

The full name of ADMM is alternating direction method of multipliers. This code implements ADMM method for solving the VRPTW problem from C101.100 dataset.

# Solution result

![](https://github.com/marcolee19970823/ADMM_VRP/blob/main/%E3%80%90ADMM%E3%80%91Solomon%20C101.100/output/fig_local_gap.svg)

Figure 1. The local bound value versus the number of iterations.

![](https://github.com/marcolee19970823/ADMM_VRP/blob/main/%E3%80%90ADMM%E3%80%91Solomon%20C101.100/output/fig_global_gap.svg)

Figure 2. The global bound value versus the number of iterations.

![](https://github.com/marcolee19970823/ADMM_VRP/blob/main/%E3%80%90ADMM%E3%80%91Solomon%20C101.100/output/fig_path.svg)

Figure 3. The path finding results for each vehicle.

# Main reference:

Yao Y, Zhu X, Dong H, et al. ADMM-based problem decomposition scheme for vehicle routing problem with time windows[J]. Transportation Research Part B: Methodological, 2019, 129: 156-174.
