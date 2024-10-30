import numpy as np
from skfuzzy import control as ctrl
from skfuzzy import membership as mf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

speed = ctrl.Antecedent(np.arange(0, 85, 0.1), 'speed')
distance = ctrl.Antecedent(np.arange(0, 3000, 0.1), 'distance')
brake = ctrl.Consequent(np.arange(0, 100, 0.1), 'brake')
throttle = ctrl.Consequent(np.arange(0, 100, 0.1), 'throttle')

speed['stopped'] = mf.trimf(speed.universe, [0, 0, 2])
speed['very slow'] = mf.trimf(speed.universe, [1, 2.5, 4])
speed['slow'] = mf.trimf(speed.universe, [2.5, 6.5, 10.5])
speed['medium fast'] = mf.trimf(speed.universe, [6.5, 26.5, 46.5])
speed['fast'] = mf.trimf(speed.universe, [26.5, 70, 85])

distance['at'] = mf.trimf(distance.universe, [0, 0, 2])
distance['very near'] = mf.trimf(distance.universe, [1, 3, 4])
distance['near'] = mf.trimf(distance.universe, [3, 101.5, 200])
distance['medium far'] = mf.trimf(distance.universe, [100, 1550, 3000])
distance['far'] = mf.trimf(distance.universe, [1500, 2250, 3000])

brake['no'] = mf.trimf(brake.universe, [0, 0, 40])
brake['very slight'] = mf.trimf(brake.universe, [20, 50, 80])
brake['slight'] = mf.trimf(brake.universe, [70, 83.5, 97])
brake['medium'] = mf.trimf(brake.universe, [95, 97, 99])
brake['full'] = mf.trimf(brake.universe, [98, 99, 100])

throttle['no'] = mf.trimf(throttle.universe, [0, 1, 2])
throttle['very slight'] = mf.trimf(throttle.universe, [1, 3, 5])
throttle['slight'] = mf.trimf(throttle.universe, [3, 16.5, 30])
throttle['medium'] = mf.trimf(throttle.universe, [20, 50, 80])
throttle['full'] = mf.trimf(throttle.universe, [60, 80, 100])


rule1 = ctrl.Rule(distance['at'] & speed['stopped'], (brake['full'], throttle['no']))
rule2 = ctrl.Rule(distance['at'] & speed['very slow'], (brake['full'], throttle['no']))
rule3 = ctrl.Rule(distance['at'] & speed['slow'], (brake['full'], throttle['no']))
rule4 = ctrl.Rule(distance['very near'] & speed['stopped'], (brake['full'], throttle['very slight']))
rule5 = ctrl.Rule(distance['very near'] & speed['very slow'], (brake['medium'], throttle['very slight']))
rule6 = ctrl.Rule(distance['very near'] & speed['slow'], (brake['medium'], throttle['very slight']))
rule7 = ctrl.Rule(distance['near'] & speed['very slow'], (brake['slight'], throttle['very slight']))
rule8 = ctrl.Rule(distance['near'] & speed['slow'], (brake['very slight'], throttle['slight']))
rule9 = ctrl.Rule(distance['medium far'] & speed['medium fast'], (brake['very slight'], throttle['medium']))
rule10 = ctrl.Rule(distance['medium far'] & speed['fast'], (brake['very slight'], throttle['medium']))
rule11 = ctrl.Rule(distance['far'] & speed['medium fast'], (brake['no'], throttle['full']))
rule12 = ctrl.Rule(distance['far'] & speed['fast'], (brake['no'], throttle['full']))

rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12]

train_ctrl = ctrl.ControlSystem(rules=rules)

train = ctrl.ControlSystemSimulation(control_system=train_ctrl)

# define the values for the inputs
#train.input['speed'] = 30
#train.input['distance'] = 2000

# compute the outputs
#train.compute()

# print the output values
#print(train.output)

# to extract one of the outputs
#print(train.output['brake'])

brake.view(sim=train)
throttle.view(sim=train)

x, y = np.meshgrid(np.linspace(speed.universe.min(), speed.universe.max(), 100),
                   np.linspace(distance.universe.min(), distance.universe.max(), 100))
z_brake = np.zeros_like(x, dtype=float)
z_throttle = np.zeros_like(x, dtype=float)

for i,r in enumerate(x):
  for j,c in enumerate(r):
    train.input['speed'] = x[i,j]
    train.input['distance'] = y[i,j]
    try:
      train.compute()
      z_brake[i,j] = train.output['brake']
      z_throttle[i,j] = train.output['throttle']
    except:
      z_brake[i,j] = float('inf')
      z_throttle[i,j] = float('inf')
    

def plot3d(x,y,z):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', linewidth=0.4, antialiased=True)

  ax.contourf(x, y, z, zdir='z', offset=-2.5, cmap='viridis', alpha=0.5)
  ax.contourf(x, y, z, zdir='x', offset=x.max()*1.5, cmap='viridis', alpha=0.5)
  ax.contourf(x, y, z, zdir='y', offset=y.max()*1.5, cmap='viridis', alpha=0.5)

  ax.view_init(30, 200)

plot3d(x, y, z_brake)
plot3d(x, y, z_throttle)