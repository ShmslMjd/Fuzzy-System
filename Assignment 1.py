import numpy as np
from skfuzzy import control as ctrl
from skfuzzy import membership as mf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the inputs
traffic_volume = ctrl.Antecedent(np.arange(0, 1200, 1), 'traffic_volume')
public_transport_usage = ctrl.Antecedent(np.arange(0, 101, 1), 'public_transport_usage')
carbon_emissions = ctrl.Antecedent(np.arange(0, 401, 1), 'carbon_emissions')

# Define the output
sustainable_transport_index = ctrl.Consequent(np.arange(0, 101, 1), 'sustainable_transport_index')

# Membership functions for Traffic Volume
traffic_volume['low'] = mf.trimf(traffic_volume.universe, [0, 0, 200])
traffic_volume['moderate'] = mf.trimf(traffic_volume.universe, [200, 500, 800])
traffic_volume['high'] = mf.trimf(traffic_volume.universe, [800, 1000, 1200])

traffic_volume.view()

# Membership functions for Public Transport Usage Rate
public_transport_usage['low'] = mf.trimf(public_transport_usage.universe, [0, 0, 20])
public_transport_usage['moderate'] = mf.trimf(public_transport_usage.universe, [20, 40, 60])
public_transport_usage['high'] = mf.trimf(public_transport_usage.universe, [60, 80, 100])

public_transport_usage.view()

# Membership functions for Carbon Emissions per Vehicle
carbon_emissions['low'] = mf.trimf(carbon_emissions.universe, [0, 0, 100])
carbon_emissions['moderate'] = mf.trimf(carbon_emissions.universe, [100, 150, 200])
carbon_emissions['high'] = mf.trimf(carbon_emissions.universe, [200, 300, 400])

carbon_emissions.view()

# Membership functions for Sustainable Transportation Index
sustainable_transport_index['low'] = mf.trimf(sustainable_transport_index.universe, [0, 0, 50])
sustainable_transport_index['moderate'] = mf.trimf(sustainable_transport_index.universe, [50, 75, 85])
sustainable_transport_index['high'] = mf.trimf(sustainable_transport_index.universe, [85, 90, 100])

sustainable_transport_index.view()

# Define the rules
rule1 = ctrl.Rule(traffic_volume['low'] & public_transport_usage['high'] & carbon_emissions['low'], sustainable_transport_index['high'])
rule2 = ctrl.Rule(traffic_volume['low'] & public_transport_usage['moderate'] & carbon_emissions['low'], sustainable_transport_index['moderate'])
rule3 = ctrl.Rule(traffic_volume['low'] & public_transport_usage['low'] & carbon_emissions['low'], sustainable_transport_index['low'])

rule4 = ctrl.Rule(traffic_volume['moderate'] & public_transport_usage['high'] & carbon_emissions['low'], sustainable_transport_index['high'])
rule5 = ctrl.Rule(traffic_volume['moderate'] & public_transport_usage['moderate'] & carbon_emissions['low'], sustainable_transport_index['moderate'])
rule6 = ctrl.Rule(traffic_volume['moderate'] & public_transport_usage['low'] & carbon_emissions['low'], sustainable_transport_index['low'])

rule7 = ctrl.Rule(traffic_volume['high'] & public_transport_usage['high'] & carbon_emissions['low'], sustainable_transport_index['moderate'])
rule8 = ctrl.Rule(traffic_volume['high'] & public_transport_usage['moderate'] & carbon_emissions['low'], sustainable_transport_index['low'])
rule9 = ctrl.Rule(traffic_volume['high'] & public_transport_usage['low'] & carbon_emissions['high'], sustainable_transport_index['low'])

# Combine the rules
rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9]

# Create the control system
sustainability_ctrl = ctrl.ControlSystem(rules)
sustainability = ctrl.ControlSystemSimulation(sustainability_ctrl)

sustainability.input['traffic_volume'] = 400
sustainability.input['public_transport_usage'] = 30
sustainability.input['carbon_emissions'] = 130

sustainability.compute()

print(sustainability.output)
print(sustainability.output['sustainable_transport_index'])

sustainable_transport_index.view(sim=sustainability)

# View the control/output space
x, y = np.meshgrid(np.linspace(traffic_volume.universe.min(), traffic_volume.universe.max(), 100),
                   np.linspace(public_transport_usage.universe.min(), public_transport_usage.universe.max(), 100))
z_sustainable_transport = np.zeros_like(x, dtype=float)

# Loop through every point and identify the value of sustainable_transport_index
for i, r in enumerate(x):
    for j, c in enumerate(r):
        sustainability.input['traffic_volume'] = x[i, j]
        sustainability.input['public_transport_usage'] = y[i, j]
        try:
            sustainability.compute()
            z_sustainable_transport[i, j] = sustainability.output['sustainable_transport_index']
        except:
            z_sustainable_transport[i, j] = float('inf')
            

# Plot the result in a 3D graph
def plot3d(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x.flatten(), y.flatten(), z.flatten(), cmap='viridis', linewidth=0.2, antialiased=True)
    ax.view_init(30, 200)

plot3d(x, y, z_sustainable_transport)
