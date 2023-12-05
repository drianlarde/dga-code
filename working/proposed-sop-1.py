import pandas as pd
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt

# Load the CSV file containing the fitness scores of the entire population for each generation
# from the proposed genetic algorithm
proposed_population_fitness_data = pd.read_csv('proposed_population_fitness_scores.csv', header=None)

# Calculate Shannon Entropy and Population Variance for each generation

# Function to calculate Shannon Entropy
def shannon_entropy(values):
    value_counts = values.value_counts()
    probabilities = value_counts / len(values)
    return -sum(probabilities * np.log2(probabilities))

# Calculate Shannon Entropy and Variance for each generation
proposed_entropy_values = proposed_population_fitness_data.apply(shannon_entropy, axis=1)
proposed_variance_values = proposed_population_fitness_data.var(axis=1)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(proposed_entropy_values, label='Shannon Entropy')
plt.plot(proposed_variance_values, label='Population Variance')
plt.xlabel('Generation')
plt.ylabel('Metric Value')
plt.title('Diversity Metrics Across Generations (Proposed GA)')
plt.legend()
plt.show()
