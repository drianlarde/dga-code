def calculate_percentage_improvement(proposed_value, modified_value):
    percentage_improvement = ((proposed_value - modified_value)/modified_value) * 100

    return percentage_improvement

# Example usage
modified_value = 1410
proposed_value = 1722

print(f"Modified Value: {modified_value}")
print(f"Proposed Value: {proposed_value}")

percentage_improvement = calculate_percentage_improvement(proposed_value, modified_value)
print(f"Percentage Improvement: {percentage_improvement:.2f}%")