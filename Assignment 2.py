import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
url = "houses.csv"
data = pd.read_csv(url)

# Inspect the dataset
print(data.head())

# Clean the dataset: Remove rows with non-numeric values in the price column
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# Check the price column before cleaning
print("Price column before cleaning:")
print(data['price'].head())

data = data[data['price'].apply(lambda x: is_number(str(x).replace(',', '')))]
data['price'] = data['price'].astype(float)

# Check the shape of the data after cleaning
print(f"Data shape after cleaning: {data.shape}")

# Optionally, drop any non-numeric columns or columns with too many missing values
data = data.select_dtypes(include=[np.number])

# Check the shape again after selecting numeric types
print(f"Data shape after selecting numeric types: {data.shape}")

# Assume 'price' is the target variable and the rest are features
X = data.drop('price', axis=1).values
y = data['price'].values

# Check the shapes of X and y
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the neural network architecture
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))  # Output layer for regression

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {test_mae}")

# Make predictions
y_pred = model.predict(X_test)

# Optional: Plot the results
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()

import random
from deap import base, creator, tools, algorithms
import tensorflow.keras.backend as K

# Define the fitness function
def evaluate(individual):
    model = Sequential()
    model.add(Dense(individual[0], input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(individual[1], activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    
    K.clear_session()
    return mae,

# Genetic Algorithm setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

toolbox.register("attr_int", random.randint, 10, 100)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

population = toolbox.population(n=10)
algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=True)

import random
from deap import base, creator, tools, algorithms
import tensorflow.keras.backend as K

# Define the fitness function
def evaluate(individual):
    model = Sequential()
    model.add(Dense(individual[0], input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(individual[1], activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    
    K.clear_session()
    return mae,

# Genetic Algorithm setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

toolbox.register("attr_int", random.randint, 10, 100)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

population = toolbox.population(n=10)
algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=True)
