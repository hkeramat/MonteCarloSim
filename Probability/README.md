### Question) The probability of rain on a given calendar day in Vancouver is p[i], where i is the day's index. For example, p[0] is the probability of rain on January 1st, and p[10] is the probability of precipitation on January 11th. Assume the year has 365 days (i.e. p has 365 elements). What is the chance it rains more than n (e.g. 100) days in Vancouver? Write a function that accepts p (probabilities of rain on a given calendar day) and n as input arguments and returns the possibility of raining at least n days.

### Solution)

```diff
def prob_rain_more_than_n(p, n):

# List of probabilities of raining at least j days (j <=n)
    probAtLeast = [1] + [0] * n

    for i in range(365):

        for j in range(min(i + 1, n), 0, -1):
# P(j rainy days today) = P(not rain today) * P(j rainy days until yesterday) + P(rain today) * P( j - 1 rainy days until yesterday)
            probAtLeast[j] =   (1 - p[i]) * probAtLeast[j] + p[i] * probAtLeast[j - 1] 

    return probAtLeast[n]
```


### State Transition Description 

The table below provides a description of the state transitions on a daily basis and how it updates the cumulative probability array `probAtLeast`:

| Component                | Description                                                                                       |
|--------------------------|---------------------------------------------------------------------------------------------------|
| **Current State**        |  `probAtLeast` is a list that contains the probability of having j rainy days until today |
| **Transition Probability** | Two transition probabilities: <br>1. **Today is not rainy**:  \( 1 - p[i] \) <br>2. **Today is rainy**: p[i]  |
| **New State**            |   `probAtLeast[j]` will be updated for both consideration that can cause j rainy days: Today not rainy and it rained j times until yesterday + today is rainy and it rained j -1 times until yesterday  |

### Complexity

- **Time Complexity**:  O(365 * n) : Iterates over each day of the year and updating comulative probabilities for up to `n` rainy days
- **Space Complexity**: O(n): An array of probabilities  for up to `n` rainy days


