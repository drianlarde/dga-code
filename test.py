import numpy as np

# This is our 'map' to find the treasure. It tells us how good a spot is.
def treasure_map(x):
    return x**2

# This is our treasure hunting adventure.
def treasure_hunt(map_function, bounds, steps=10):
    # We have a team of 5 friends searching for the treasure.
    team_size = 5

    # Each friend picks a random spot to start looking within the bounds.
    team = np.random.uniform(bounds[0], bounds[1], team_size)

    for step in range(steps):
        # Each friend thinks of a new spot near their current spot.
        new_spots = team + np.random.uniform(-1, 1, team_size)

        # They check the new spot to see if it's closer to the treasure.
        for i in range(team_size):
            if map_function(new_spots[i]) < map_function(team[i]):
                # If the new spot is better, they move there.
                team[i] = new_spots[i]

    # After all the steps, the friend closest to the treasure is the winner!
    best_spot = team[np.argmin([map_function(spot) for spot in team])]
    return best_spot

# The area where the treasure could be.
bounds = (-5, 5)

# Let's go on the treasure hunt!
best_spot_found = treasure_hunt(treasure_map, bounds)
print("Best spot for the treasure:", best_spot_found)
