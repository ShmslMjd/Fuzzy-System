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

# Membership functions for Public Transport Usage Rate
public_transport_usage['low'] = mf.trimf(public_transport_usage.universe, [0, 0, 20])
public_transport_usage['moderate'] = mf.trimf(public_transport_usage.universe, [20, 40, 60])
public_transport_usage['high'] = mf.trimf(public_transport_usage.universe, [60, 80, 100])

# Membership functions for Carbon Emissions per Vehicle
carbon_emissions['low'] = mf.trimf(carbon_emissions.universe, [0, 0, 100])
carbon_emissions['moderate'] = mf.trimf(carbon_emissions.universe, [100, 150, 200])
carbon_emissions['high'] = mf.trimf(carbon_emissions.universe, [200, 300, 400])

# Membership functions for Sustainable Transportation Index
sustainable_transport_index['low'] = mf.trimf(sustainable_transport_index.universe, [0, 0, 50])
sustainable_transport_index['moderate'] = mf.trimf(sustainable_transport_index.universe, [50, 75, 85])
sustainable_transport_index['high'] = mf.trimf(sustainable_transport_index.universe, [85, 90, 100])

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
sustainability_ctrl = ctrl.ControlSystem(rules=rules)
sustainability = ctrl.ControlSystemSimulation(control_system=sustainability_ctrl)

# Define the values for the inputs for testing
sustainability.input['traffic_volume'] = 500
sustainability.input['public_transport_usage'] = 30
sustainability.input['carbon_emissions'] = 150

# Compute the outputs
sustainability.compute()

# Print the output values
print(sustainability.output['sustainable_transport_index'])

# To visualize the outputs
traffic_volume_range = np.linspace(0, 1200, 10)  # Reduced number for quicker computation
public_transport_usage_range = np.linspace(0, 100, 10)
carbon_emissions_range = np.linspace(0, 400, 10)

# Prepare an array to hold the outputs
sustainable_transport_index_output = []

# Loop through the combinations of inputs
for traffic in traffic_volume_range:
    for public in public_transport_usage_range:
        for carbon in carbon_emissions_range:
            sustainability.input['traffic_volume'] = traffic
            sustainability.input['public_transport_usage'] = public
            sustainability.input['carbon_emissions'] = carbon
            sustainability.compute()
            sustainable_transport_index_output.append(sustainability.output['sustainable_transport_index'])

# Convert the output list to a NumPy array for easier manipulation
sustainable_transport_index_output = np.array(sustainable_transport_index_output)

# Reshape the output for plotting
sustainable_transport_index_output = sustainable_transport_index_output.reshape(len(traffic_volume_range), len(public_transport_usage_range), len(carbon_emissions_range))

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a meshgrid for plotting
X, Y = np.meshgrid(public_transport_usage_range, traffic_volume_range)
Z = sustainable_transport_index_output[:, :, 0]  # Use the first carbon emissions value for simplicity

# Plot the surface
ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_xlabel('Public Transport Usage Rate')
ax.set_ylabel('Traffic Volume')
ax.set_zlabel('Sustainable Transportation Index')
plt.show()
