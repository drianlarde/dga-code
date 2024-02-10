import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('best_fitness_scores.csv')

# Define a function to plot the chart and check for plateaus for each column
def plot_fitness_scores(generations, fitness_scores, column_name):
    # Calculate the absolute change in fitness score
    fitness_changes = fitness_scores.diff().abs().fillna(0)

    # Normalize the changes
    fitness_changes_normalized = fitness_changes / fitness_changes.max()

    # Define a threshold for change to consider it a plateau
    plateau_threshold = fitness_changes_normalized.mean() * 0.1

    # Calculate percentages for significant changes and plateaus
    significant_changes = fitness_changes_normalized > plateau_threshold
    plateaus = fitness_changes_normalized <= plateau_threshold
    percent_significant_changes = np.sum(significant_changes) / len(generations) * 100
    percent_plateaus = np.sum(plateaus) / len(generations) * 100

    # Plotting the fitness scores over generations with change volume bars
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot the line of fitness scores
    ax1.plot(generations, fitness_scores, label='Fitness Score', color='black', linewidth=2)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness Score', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Add bars for significant changes
    ax1.bar(generations[significant_changes], fitness_scores[significant_changes], color='green', label='Significant Change', alpha=0.4, width=1)

    # Add bars for plateaus (low changes)
    ax1.bar(generations[plateaus], fitness_scores[plateaus], color='orange', label='Plateau', alpha=0.4, width=1)

    # Add a legend
    ax1.legend(loc='upper left')

    # Display percentages on the chart
    plt.title(f'Fitness Score and Change Volume over Generations for {column_name}\n{percent_significant_changes:.2f}% Significant Changes, {percent_plateaus:.2f}% Plateaus')
    plt.show()

# Assuming the first column is 'Generation' and it's numeric and sequential
generations = np.arange(len(df))

# Iterate through each column in the DataFrame, excluding 'Generation' if present
for column_name in df.columns:
    fitness_scores = df[column_name]
    plot_fitness_scores(generations, fitness_scores, column_name)
