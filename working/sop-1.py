import pandas as pd
import numpy as np
from scipy.stats import entropy

# Load the fitness scores data
data = pd.read_csv('population_fitness_scores.csv', header=None)

# Function to calculate Shannon Entropy
def shannon_entropy(values):
    _, counts = np.unique(values, return_counts=True)
    return entropy(counts)

# Calculate Shannon Entropy and Variance for each generation
entropy_values = data.apply(shannon_entropy, axis=1)
variance_values = data.var(axis=1)

# Plotting
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(entropy_values, label='Shannon Entropy')
plt.plot(variance_values, label='Population Variance')
plt.xlabel('Generation')
plt.ylabel('Metric Value')
plt.title('Diversity Metrics Across Generations')
plt.legend()
plt.show()
