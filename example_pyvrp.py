import matplotlib.pyplot as plt
from pyvrp import Model
from pyvrp.plotting import plot_coordinates, plot_solution
from pyvrp.stop import MaxRuntime

# Coordinates and Demands
COORDS = [
    (456, 320), (228, 0), (912, 0), (0, 80), (114, 80), (570, 160),
    (798, 160), (342, 240), (684, 240), (570, 400), (912, 400),
    (114, 480), (228, 480), (342, 560), (684, 560), (0, 640), (798, 640)
]

DEMANDS = [0, 1, 1, 2, 4, 2, 4, 8, 8, 1, 2, 1, 2, 4, 4, 8, 8]

# Model setup
m = Model()
m.add_vehicle_type(4, capacity=15)
depot = m.add_depot(x=COORDS[0][0], y=COORDS[0][1])
clients = [m.add_client(x=COORDS[idx][0], y=COORDS[idx][1], delivery=DEMANDS[idx]) for idx in range(1, len(COORDS))]
locations = [depot] + clients

# Add edges with Manhattan distance
for frm in locations:
    for to in locations:
        distance = abs(frm.x - to.x) + abs(frm.y - to.y)
        m.add_edge(frm, to, distance=distance)



# Solve the problem
res = m.solve(stop=MaxRuntime(1), display=True)  # one second

# Check if a valid solution was found and plot it
if res.best:
    _, ax = plt.subplots(figsize=(8, 8))
    plot_solution(res.best, m.data(), ax=ax)
    plt.title("VRP Solution - Optimized Routes")
    plt.show()
else:
    print("No feasible solution was found.")
