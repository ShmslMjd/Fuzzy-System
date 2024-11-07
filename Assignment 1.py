import numpy as np
from skfuzzy import control as ctrl
from skfuzzy import membership as mf

# Define new linguistic variables with appropriate ranges
traffic_volume = ctrl.Antecedent(np.arange(0, 3000, 0.1), 'traffic volume')
public_transport_usage = ctrl.Antecedent(np.arange(0, 100, 0.1), 'public transport usage')
API = ctrl.Consequent(np.arange(0, 500, 0.1), 'API')
sustainability_index = ctrl.Consequent(np.arange(0, 100, 0.1), 'sustainability index')

# Membership functions for traffic volume based on your points
traffic_volume['very low'] = mf.trimf(traffic_volume.universe, [0, 0, 400])
traffic_volume['low'] = mf.trimf(traffic_volume.universe, [200, 700, 1200])
traffic_volume['medium'] = mf.trimf(traffic_volume.universe, [1000, 1500, 2000])
traffic_volume['high'] = mf.trapmf(traffic_volume.universe, [1500, 2000, 3000, 3000])

traffic_volume.view()

# Membership functions for public transport usage
public_transport_usage['very low'] = mf.trimf(public_transport_usage.universe, [0, 0, 20])
public_transport_usage['low'] = mf.trimf(public_transport_usage.universe, [10, 30, 50])
public_transport_usage['medium'] = mf.trimf(public_transport_usage.universe, [40, 60, 80])
public_transport_usage['high'] = mf.trimf(public_transport_usage.universe, [70, 90, 100])
public_transport_usage['very high'] = mf.trapmf(public_transport_usage.universe, [80, 90, 100, 100])

public_transport_usage.view()

# Membership functions for API
API['very low'] = mf.trimf(API.universe, [0, 0, 100])
API['low'] = mf.trimf(API.universe, [50, 150, 250])
API['medium'] = mf.trimf(API.universe, [200, 300, 400])
API['high'] = mf.trimf(API.universe, [350, 400, 450])
API['very high'] = mf.trimf(API.universe, [400, 500, 500])

API.view()

# Membership functions for sustainability index
sustainability_index['very low'] = mf.trimf(sustainability_index.universe, [0, 0, 20])
sustainability_index['low'] = mf.trimf(sustainability_index.universe, [10, 30, 50])
sustainability_index['medium'] = mf.trimf(sustainability_index.universe, [40, 60, 80])
sustainability_index['high'] = mf.trimf(sustainability_index.universe, [70, 90, 100])
sustainability_index['very high'] = mf.trapmf(sustainability_index.universe, [80, 90, 100, 100])

sustainability_index.view()

# Define rules 
rule1 = ctrl.Rule(public_transport_usage['very low'] & traffic_volume['very low'], (API['low'], sustainability_index['medium'])) 
rule2 = ctrl.Rule(public_transport_usage['very low'] & traffic_volume['low'], (API['medium'], sustainability_index['low'])) 
rule3 = ctrl.Rule(public_transport_usage['very low'] & traffic_volume['medium'], (API['high'], sustainability_index['low'])) 
rule4 = ctrl.Rule(public_transport_usage['very low'] & traffic_volume['high'], (API['very high'], sustainability_index['very low'])) 
rule5 = ctrl.Rule(public_transport_usage['low'] & traffic_volume['very low'], (API['very low'], sustainability_index['high'])) 
rule6 = ctrl.Rule(public_transport_usage['low'] & traffic_volume['low'], (API['low'], sustainability_index['medium'])) 
rule7 = ctrl.Rule(public_transport_usage['low'] & traffic_volume['medium'], (API['medium'], sustainability_index['low'])) 
rule8 = ctrl.Rule(public_transport_usage['low'] & traffic_volume['high'], (API['high'], sustainability_index['low'])) 
rule9 = ctrl.Rule(public_transport_usage['medium'] & traffic_volume['very low'], (API['very low'], sustainability_index['very high'])) 
rule10 = ctrl.Rule(public_transport_usage['medium'] & traffic_volume['low'], (API['low'], sustainability_index['high'])) 
rule11 = ctrl.Rule(public_transport_usage['medium'] & traffic_volume['medium'], (API['medium'], sustainability_index['medium'])) 
rule12 = ctrl.Rule(public_transport_usage['medium'] & traffic_volume['high'], (API['high'], sustainability_index['low'])) 
rule13 = ctrl.Rule(public_transport_usage['high'] & traffic_volume['very low'], (API['very low'], sustainability_index['very high'])) 
rule14 = ctrl.Rule(public_transport_usage['high'] & traffic_volume['low'], (API['low'], sustainability_index['high'])) 
rule15 = ctrl.Rule(public_transport_usage['high'] & traffic_volume['medium'], (API['medium'], sustainability_index['medium'])) 
rule16 = ctrl.Rule(public_transport_usage['high'] & traffic_volume['high'], (API['high'], sustainability_index['medium'])) 
rule17 = ctrl.Rule(public_transport_usage['very high'] & traffic_volume['very low'], (API['very low'], sustainability_index['very high'])) 
rule18 = ctrl.Rule(public_transport_usage['very high'] & traffic_volume['low'], (API['low'], sustainability_index['high'])) 
rule19 = ctrl.Rule(public_transport_usage['very high'] & traffic_volume['medium'], (API['medium'], sustainability_index['high'])) 
rule20 = ctrl.Rule(public_transport_usage['very high'] & traffic_volume['high'], (API['very high'], sustainability_index['medium'])) 

rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20]
# Control system and simulation
control_system = ctrl.ControlSystem(rules=rules)
simulation = ctrl.ControlSystemSimulation(control_system=control_system)

# Test input values
simulation.input['traffic volume'] = 1500
simulation.input['public transport usage'] = 50

simulation.compute()

print(simulation.output)
print(simulation.output['API'])

API.view(sim=simulation)
sustainability_index.view(sim=simulation)

# Generate 3D plots
x, y = np.meshgrid(np.linspace(traffic_volume.universe.min(), traffic_volume.universe.max(), 100),
                   np.linspace(public_transport_usage.universe.min(), public_transport_usage.universe.max(), 100))
z_API = np.zeros_like(x, dtype=float)
z_sustainability_index = np.zeros_like(x, dtype=float)

for i, r in enumerate(x):
    for j, c in enumerate(r):
        simulation.input['traffic volume'] = x[i, j]
        simulation.input['public transport usage'] = y[i, j]
        try:
            simulation.compute()
            z_API[i, j] = simulation.output['API']
            z_sustainability_index[i, j] = simulation.output['sustainability index']
        except:
            z_API[i, j] = float('inf')
            z_sustainability_index[i, j] = float('inf')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot3d(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', linewidth=0.4, antialiased=True)
    ax.contourf(x, y, z, zdir='z', offset=-2.5, cmap='viridis', alpha=0.5)
    ax.contourf(x, y, z, zdir='x', offset=x.max()*1.5, cmap='viridis', alpha=0.5)
    ax.contourf(x, y, z, zdir='y', offset=y.max()*1.5, cmap='viridis', alpha=0.5)
    ax.view_init(30, 200)

plot3d(x, y, z_API)
plot3d(x, y, z_sustainability_index)

plt.show()
