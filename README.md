# MonteCarloSim
#### If I play a game with the probability of wining equal to 0.2, each rounds of game ends when I loose two times in a row, what is the expected of rounds that I play in this game?
```
import random

def play_game():
    """Play one round of the game and return the number of rounds played."""
    num_rounds = 0
    num_losses_in_a_row = 0
    while num_losses_in_a_row < 2:
        if random.random() < 0.2:
            num_losses_in_a_row = 0
        else:
            num_losses_in_a_row += 1
        num_rounds += 1
    return num_rounds

num_simulations = 1000000  # number of simulations to run
total_rounds_played = 0
for _ in range(num_simulations):
    total_rounds_played += play_game()
expected_rounds = total_rounds_played / num_simulations
print("Expected number of rounds: ", expected_rounds)
```

### Monte Carlo simulation to estimate the probability of throwing a dart inside a circle inscribed in a square:
```
import random

def throw_dart():
    """Throw one dart and return True if it lands inside the circle, False otherwise."""
    x = random.uniform(-1, 1)  # generate random x coordinate between -1 and 1
    y = random.uniform(-1, 1)  # generate random y coordinate between -1 and 1
    distance_squared = x**2 + y**2
    if distance_squared <= 1:
        return True
    else:
        return False

num_simulations = 1000000  # number of simulations to run
num_darts_inside_circle = 0
for _ in range(num_simulations):
    if throw_dart():
        num_darts_inside_circle += 1
estimated_probability = num_darts_inside_circle / num_simulations
print("Estimated probability: ", estimated_probability)

```
