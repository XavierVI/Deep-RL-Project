### About
This repository is an extension of an assignment from a class. It implements multiple deep reinforcement learning algorithms, and measures their performance on Bipedal Walker.


### Global Config file and Training
The entries

```
"learning_rate": 0.001,
"gamma": 0.99,
```

are only used when running the python scripts without `--parallel`. If `--parallel` is given, it will try to run all possible combinations of 

```
"learning_rates": [0.0001, 0.001, 0.01],
"gammas": [0.90, 0.95, 0.99],
"boltzmann_options": [true, false]
```

using a process pool, so multiple agents can be trained at the same time.

I added additional parameters to `global_config.json` because I wanted to fine tune epsilon-greedy and Boltzmann sampling as much as possible to find the best one for each algorithm.


### Plotting
A moving average with a window size of `N` can actually be computed using a convolution between the rewards and an array of ones divided by the window size (5).


### Max Timesteps
Setting the maximum number of time steps to 1000 did lead to really high rewards, but this would slow down training significantly. 500 seems to be yielding some decent results.


### References
1. [Gymnasium Docs](https://gymnasium.farama.org/)
2. [Wikipedia: Moving Average]()
3. [NumPy: ufunc.at (used for computing averages in plot.py)](https://numpy.org/devdocs//reference/generated/numpy.ufunc.at.html)
4. [NumPy: Convolve](https://numpy.org/doc/stable/reference/generated/numpy.convolve.html)
5. [Water Programming: Implementation of the Moving Average Filter Using Convolution](https://waterprogramming.wpcomstaging.com/2018/09/04/implementation-of-the-moving-average-filter-using-convolution/)
6. [VizDoom Repository](https://github.com/Farama-Foundation/ViZDoom)
7. [VizDoom Documentation](https://vizdoom.cs.put.edu.pl/)