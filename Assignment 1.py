import numpy as np
from skfuzzy import control as ctrl
from skfuzzy import membership as mf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the fuzzy variables
traffic_volume = ctrl.Antecedent(np.arange(0, 1201, 1), 'traffic_volume')
public_transport_usage = ctrl.Antecedent(np.arange(0, 101, 1), 'public_transport_usage')
carbon_emissions = ctrl.Antecedent(np.arange(0, 401, 1), 'carbon_emissions')
sustainable_transportation_index = ctrl.Consequent(np.arange(0, 101, 1), 'sustainable_transportation_index')

# Define membership functions for Traffic Volume
traffic_volume['very low'] = mf.trimf(traffic_volume.universe, [0, 0, 200])
traffic_volume['low'] = mf.trimf(traffic_volume.universe, [100, 200, 400])
traffic_volume['moderate'] = mf.trimf(traffic_volume.universe, [300, 500, 700])
traffic_volume['high'] = mf.trimf(traffic_volume.universe, [600, 800, 1000])
traffic_volume['very high'] = mf.trimf(traffic_volume.universe, [900, 1200, 1200])

# Define membership functions for Public Transport Usage Rate
public_transport_usage['very low'] = mf.trimf(public_transport_usage.universe, [0, 0, 20])
public_transport_usage['low'] = mf.trimf(public_transport_usage.universe, [10, 20, 40])
public_transport_usage['moderate'] = mf.trimf(public_transport_usage.universe, [30, 50, 70])
public_transport_usage['high'] = mf.trimf(public_transport_usage.universe, [60, 80, 90])
public_transport_usage['very high'] = mf.trimf(public_transport_usage.universe, [80, 100, 100])

# Define membership functions for Carbon Emissions per Vehicle
carbon_emissions['very low'] = mf.trimf(carbon_emissions.universe, [0, 0, 100])
carbon_emissions['low'] = mf.trimf(carbon_emissions.universe, [50, 100, 150])
carbon_emissions['moderate'] = mf.trimf(carbon_emissions.universe, [100, 200, 300])
carbon_emissions['high'] = mf.trimf(carbon_emissions.universe, [200, 300, 350])
carbon_emissions['very high'] = mf.trimf(carbon_emissions.universe, [300, 400, 400])

# Define membership functions for Sustainable Transportation Index
sustainable_transportation_index['very low'] = mf.trimf(sustainable_transportation_index.universe, [0, 0, 25])
sustainable_transportation_index['low'] = mf.trimf(sustainable_transportation_index.universe, [15, 35, 55])
sustainable_transportation_index['moderate'] = mf.trimf(sustainable_transportation_index.universe, [45, 60, 75])
sustainable_transportation_index['high'] = mf.trimf(sustainable_transportation_index.universe, [70, 85, 90])
sustainable_transportation_index['very high'] = mf.trimf(sustainable_transportation_index.universe, [85, 95, 100])

# View the membership functions (optional)
traffic_volume.view()
public_transport_usage.view()
carbon_emissions.view()
sustainable_transportation_index.view()

# Define fuzzy rules with expanded membership functions
rule1 = ctrl.Rule(traffic_volume['very low'] & public_transport_usage['very high'] & carbon_emissions['very low'],
                  sustainable_transportation_index['very high'])
rule2 = ctrl.Rule(traffic_volume['very high'] & public_transport_usage['very low'] & carbon_emissions['very high'],
                  sustainable_transportation_index['very low'])
rule3 = ctrl.Rule(traffic_volume['moderate'] & public_transport_usage['moderate'] & carbon_emissions['moderate'],
                  sustainable_transportation_index['moderate'])
rule4 = ctrl.Rule(traffic_volume['high'] & public_transport_usage['high'] & carbon_emissions['low'],
                  sustainable_transportation_index['high'])
rule5 = ctrl.Rule(traffic_volume['low'] & public_transport_usage['low'] & carbon_emissions['low'],
                  sustainable_transportation_index['low'])

# Define the control system with the rules
sustainability_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
sustainability = ctrl.ControlSystemSimulation(sustainability_ctrl)

# Input values for a test scenario
sustainability.input['traffic_volume'] = 300
sustainability.input['public_transport_usage'] = 50
sustainability.input['carbon_emissions'] = 120

# Perform the computation
sustainability.compute()

print(sustainability.output)
print(sustainability.output['sustainable_transportation_index'])

# Visualize the output for Sustainable Transportation Index
sustainable_transportation_index.view(sim=sustainability)

# Create 3D plots for Surface Visualization
x, y = np.meshgrid(np.linspace(traffic_volume.universe.min(), traffic_volume.universe.max(), 50),
                   np.linspace(public_transport_usage.universe.min(), public_transport_usage.universe.max(), 50),
                   np.linspace(carbon_emissions.universe.min(), carbon_emissions.universe.max(), 50))
z_sustainability = np.zeros_like(x, dtype=float)

for i,r in enumerate(x):
  for j,c in enumerate(r):
        sustainability.input['traffic_volume'] = x[i, j]
        sustainability.input['public_transport_usage'] = y[i, j]
        sustainability.input['carbon_emissions'] = 150  # keeping carbon_emissions constant for this visualization
        try:
          sustainability.compute()
          z_sustainability[i, j] = sustainability.output['sustainable_transportation_index']
        except:
            z_sustainability[i,j] = float('inf')

def plot3d(x,y,z):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', linewidth=0.4, antialiased=True)

  ax.contourf(x, y, z, zdir='z', offset=-2.5, cmap='viridis', alpha=0.5)
  ax.contourf(x, y, z, zdir='x', offset=x.max()*1.5, cmap='viridis', alpha=0.5)
  ax.contourf(x, y, z, zdir='y', offset=y.max()*1.5, cmap='viridis', alpha=0.5)

  ax.view_init(30, 200)

plot3d(x, y, z_sustainability)
plt.show()
