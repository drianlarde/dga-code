import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the datasets
fitness_scores_df = pd.read_csv('best_fitness_scores.csv')
diversity_df = pd.read_csv('diversity_across_generations.csv')

# Calculate the mean and standard deviation of fitness scores at each generation across all configurations
mean_fitness_scores = fitness_scores_df.mean(axis=1)
std_fitness_scores = fitness_scores_df.std(axis=1)

# For diversity, we assume that each column after the 'generation' column is a separate run
mean_diversity = diversity_df.iloc[:, 1:].mean(axis=1)
std_diversity = diversity_df.iloc[:, 1:].std(axis=1)

# Plotting
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Plot for fitness scores
axes[0].plot(fitness_scores_df.index, mean_fitness_scores, label='Mean Fitness Score')
axes[0].fill_between(fitness_scores_df.index, mean_fitness_scores - std_fitness_scores, 
                     mean_fitness_scores + std_fitness_scores, color='gray', alpha=0.2,
                     label='Standard Deviation')

axes[0].set_title('Mean Fitness Scores with Standard Deviation Across Runs')
axes[0].set_ylabel('Fitness Score')
axes[0].legend()

# Plot for diversity
axes[1].plot(diversity_df.index, mean_diversity, label='Mean Diversity Score', color='orange')
axes[1].fill_between(diversity_df.index, mean_diversity - std_diversity, 
                     mean_diversity + std_diversity, color='gray', alpha=0.2,
                     label='Standard Deviation', linestyle='--')

axes[1].set_title('Mean Diversity Scores with Standard Deviation Across Runs')
axes[1].set_xlabel('Generations')
axes[1].set_ylabel('Diversity Score')
axes[1].legend()

plt.tight_layout()
plt.show()

# Calculating the average standard deviation across all generations for fitness and diversity
avg_std_fitness = std_fitness_scores.mean()
avg_std_diversity = std_diversity.mean()

avg_std_fitness, avg_std_diversity
