# Mancala

Mancala board game written in python.

![img](https://github.com/qqhann/Mancala/blob/main/assets/preview_cli.png)

## Installation

```shell
$ pip install poetry
$ poetry install
$ poetry update
$ poetry build
```

## Usage

### Play a game with agents

```shell
$ poetry run mancala play --player0 human --player1 random
```

### Compare each agents and plot their win rates

The values are player1's (second move) win rates in percentage

```shell
$ poetry run mancala arena
              p0_random  p0_exact  p0_max  p0_minimax  p0_negascout  p0_mcts
p1_random          50.0      53.0     3.0         0.0           0.0      6.0
p1_exact           42.0      48.0     4.0         1.0           1.0      9.0
p1_max             95.0      91.0    41.0         0.0           3.0     66.0
p1_minimax        100.0      96.0    87.0        30.0          39.0     90.0
p1_negascout      100.0      97.0    84.0        19.0          32.0     87.0
p1_mcts            94.0      96.0    39.0         3.0           7.0     45.0
```

## Algorithms

Mancala is a game with perfect information.

### Mini-Max

Mini-max is an algorithm for n-player zero-sum games.
The concept is to assume the opponent will take their best move and try to minimize them.

- MiniMax <https://en.wikipedia.org/wiki/Minimax>
- Alpha-beta pruning <https://en.wikipedia.org/wiki/Alpha–beta_pruning>
- Negamax <https://en.wikipedia.org/wiki/Negamax>
- NegaScout (PVS, principal variation search) <https://en.wikipedia.org/wiki/Principal_variation_search>
- Monte Carlo Tree Search <https://en.wikipedia.org/wiki/Monte_Carlo_tree_search>


### Policy Iteration

Using Dynamic Programming (DP), calculate value for states and memorize them.
Use the value and policy for planning.

## References

- <https://github.com/mdavolio/mancala>
- <https://github.com/qqpann/Mancala>
