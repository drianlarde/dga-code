def calculate_percentage_improvement(proposed_fitness, modified_fitness):
    percentage_improvement = ((proposed_fitness - modified_fitness) / abs(modified_fitness)) * 100
    return percentage_improvement

# Fitness scores at generation 7
modified_fitness = -750
proposed_fitness = -10

print(f"Modified Fitness at Generation 7: {modified_fitness}")
print(f"Proposed Fitness at Generation 7: {proposed_fitness}")

percentage_improvement = calculate_percentage_improvement(proposed_fitness, modified_fitness)
print(f"Percentage Improvement in Fitness at Generation 7: {percentage_improvement:.2f}%")