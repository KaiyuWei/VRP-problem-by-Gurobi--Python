import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import math
import matplotlib.pyplot as plt

# Read data
data0 = pd.read_csv("Data.tsv", delimiter="\t")

# Set nodes
xc = np.array(data0['lon'])  # x-coordinate of clients
xc = [i * 5 for i in xc]
xDepot = np.mean(xc)
xc = xc + [xDepot, xDepot]  # Convert longitude to km, including last two virtual depots' location.
yc = np.array(data0['lat'])  # y-coordinate of clients
yc = [i * 5 for i in yc]
yDepot = np.mean(yc)
yc = yc + [yDepot, yDepot]  # Convert latitude to km, including last two virtual depots' location.

# Set parameter for the vehicle
Capacity = 1000  # In the unit of liters (L)
AvgVelocity = 30  # Set average speed of the vehicle (km/h)

# Set graphical parameters of nodes
AllNodes = [i for i in range(len(xc))]   # nodes indicators. nodes n(starting) and n + 1(ending) are virtual depots.
CusNodes = AllNodes[:len(AllNodes)-2]  # Customer nodes by removing last two virtual depots.
Edges = [(i, j) for i in AllNodes for j in AllNodes if i != j]   # Edges between each pair of nodes
Distance = {(i, j): np.hypot(xc[i] - xc[j], yc[i] - yc[j])
            for i, j in Edges}  # Distances between each pair of nodes (km)

for i in range(len(AllNodes)):
    if i != len(AllNodes):
        Distance[len(AllNodes) - 2, i] = 0  # Distance of starting depot to any node is 0.

TravelTime = {(i, j): Distance[i, j] / (AvgVelocity / 3600)
              for i, j in Edges}  # Travel time between each pair of nodes (s)
PickAmount = {i: data0['size'][i] * data0['quantity'][i]
              for i in CusNodes}  # Amount to be picked from each customer

# Set time windows
ServiceTime = {i: data0['service_time'][i] for i in CusNodes}  # Set service time
time1 = data0['time_start']
time1 = [float(time1[i][0]) if time1[i][1] == ':'
         else float(time1[i][0] + time1[i][1]) for i in CusNodes]  # Convert string to float numbers
StartTime = {i: time1[i] * 3600 for i in CusNodes}  # Set starting time in units of seconds (s)
time2 = data0['time_end']
time2 = [26.0 if pd.isnull(time2[i])   # If value = nan then set the end time to a very late time.
         else float(time2[i][0]) if time2[i][1] == ':'
         else float(time2[i][0] + time2[i][1]) for i in CusNodes]
EndTime = {i: time2[i] * 3600 for i in CusNodes}  # Endtime
TimeWindow = {i: (StartTime[i], EndTime[i]) for i in CusNodes}  # Time windows for all customer nodes

# Create an optimization model
mdl = gp.Model('Seenons')

# Add decision variables
x = mdl.addVars(Edges, vtype=GRB.BINARY)  # Add decision variables x_ij
ArTime = mdl.addVars(CusNodes, vtype=GRB.CONTINUOUS)  # Arrival time at each node
DeTime = mdl.addVars(CusNodes, vtype=GRB.CONTINUOUS)  # Departure time at each node

# Add objective
mdl.modelSense = GRB.MINIMIZE
mdl.setObjective(gp.quicksum(x[i, j] * Distance[i, j] for i, j in Edges))

# Add constraints
# Constraints controlling the number of going into and out of a customer node
mdl.addConstrs(gp.quicksum(x[i, j] for j in AllNodes if j != i) == 1 for i in CusNodes)
mdl.addConstrs(gp.quicksum(x[i, j] for i in AllNodes if j != i) == 1 for j in CusNodes)
mdl.addConstr(gp.quicksum(x[(len(AllNodes) - 2), j]
                          for j in AllNodes if j != (len(AllNodes) - 2)) == 1)  # 2nd last node as virtual start depot
mdl.addConstr(gp.quicksum(x[i, (len(AllNodes) - 1)]
                          for i in AllNodes if i != (len(AllNodes) - 1)) == 1)  # Last node as virtual end depot.

# Constraints about the Time window
WorkStart = 9  # The vehicle start working time of one day
mdl.addConstrs(DeTime[i] + TravelTime[i, j] - ArTime[j] <= (1 - x[i, j]) * 90000 
               for i in CusNodes for j in CusNodes if i != j)  # 90000 as the big-M
mdl.addConstrs(DeTime[i] >= StartTime[i] + ServiceTime[i] for i in CusNodes)
mdl.addConstrs(DeTime[i] >= ArTime[i] + ServiceTime[i] for i in CusNodes)  # Constraints about the departure time.
mdl.addConstrs(ArTime[i] <= EndTime[i] for i in CusNodes)  # Constraints about the arrival time
mdl.addConstrs((x[len(AllNodes) - 2, i] == 1) >> (ArTime[i] == WorkStart * 3600)
               for i in CusNodes)   # The first nodes starts at 9:00

# Optimize
mdl.optimize()

# Out put data
SDep = len(AllNodes) - 2  # second last node in AllNodes as the start depot
EDep = len(AllNodes) - 1  # Last node in AllNodes as the end depot
ActiveEdges = [a for a in Edges if x[a].X > 0.9]  # Edges that are selected
Route = []  # A list for storing the route
sn = SDep  # Set a start node for the while loop
while sn != EDep:
    for (i, j) in ActiveEdges:
        if i == sn:
            Route = Route + [(i, j)]
            sn = j
            break
SeqNode = [i[1] for i in Route[:-1]]  # Nodes in the sequence of the route excluding two virtual nodes
Output_Ar = {i: ArTime.values()[i].X
             for i in SeqNode}  # Arrival times of each node excluding two virtual nodes.
Output_Ar = {i: '%d:%d' % (ArTime.values()[i].X // 3600, math.modf(ArTime.values()[i].X/3600)[0] * 60)
             for i in Output_Ar.keys()}  # Convert arrival time to 'hh:mm' format in order to output
Output_De = {i: DeTime.values()[i].X
             for i in SeqNode}  # Departure times of each node excluding two virtual nodes.
Output_De = {i: '%d:%d' % (DeTime.values()[i].X // 3600, math.modf(DeTime.values()[i].X/3600)[0] * 60)
             for i in Output_Ar.keys()}  # Convert arrival time to 'hh:mm' format in order to output

# Calculate the unloading time
nn = 0  # Node number for while-loop
TotLoad = 0  # Total load till the current node
Unload = {i: "No" for i in SeqNode}  # Indicates whether the vehicle should go unloading
while nn < len(SeqNode) - 1:
    TotLoad = TotLoad + PickAmount[SeqNode[nn]]  # Total load after serving current load
    NextLoad = TotLoad + PickAmount[SeqNode[nn + 1]]  # if the vehicle doesn't unload and go to the next node
    if NextLoad >= Capacity:     # "=" here for a buffer
        Unload[SeqNode[nn]] = "Yes"  # =1 means the vehicle need to go unloading after leaving current node
        TotLoad = 0  # Clear the load before going to the next load
    nn = nn + 1

# Output the result
print('Optimal route:\n')
print('Node\tArTime\tDeTime\tUnload\n')
for i in SeqNode:
    print('%d\t%s\t%s\t%s\n' % (i, Output_Ar[i], Output_De[i], Unload[i]))

# Plot the solution
for i, j in ActiveEdges:
    if i != SDep and j != SDep and i != EDep and j != EDep:
        plt.plot([xc[i], xc[j]], [yc[i], yc[j]], c='g', zorder=0)  # Plot active edges excluding two virtual depots

plt.scatter(xc[0:26], yc[0:26], c='b')  # Location of clients
plt.text(xc[SeqNode[0]], yc[SeqNode[0]],'Start point')
plt.show()
