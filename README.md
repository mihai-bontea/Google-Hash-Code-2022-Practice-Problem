# Google Hash Code 2022 Practice Problem
![Image](https://github.com/user-attachments/assets/41dec048-0b7a-42a3-8ec0-7f17874d1b54)

>Everyone has their own pizza preferences. Each of your potential clients has some ingredients they like, and maybe some ingredients they dislike. Each client will come to your pizzeria if both conditions are true:

> * all the ingredients they like are on the pizza, and

> * none of the ingredients they dislike are on the pizza

>Each client is OK with additional ingredients they neither like or dislike being present on the pizza. Your task is to choose which ingredients to put on your only pizza type, to maximize the number of clients that will visit your pizzeria.

## Solution 1

### Strategy

The problem is framed as a **binary decision tree**: at each depth *i* in the tree, we can either decide to take this ingredient or not. The solutions will be defined by the leaf nodes of the tree. For the biggest input, there's 10,000 ingredients, which means 2^10,000 solutions (astronomically huge). Thus, a **branch and bounding** technique is necessary, where only a small, promising portion of the decision tree is checked:

* if no one likes an ingredient, its addition is not attempted

* if no one dislikes an ingredient, its removal is not attempted

* if the best solution found so far has fewer clients lost than the current state, this path will be abandoned

The operation of evaluating paths in the decision tree is a great candidate for parallel processing. 8 threads are started from depth 4 in the tree, and they update a shared variable **best_simulation_state**. The simulation state of a node is mainly defined by two **bitsets**, *is_client_remaining* and *is_ingredient_chosen* which provide a memory efficient way to keep track of previous choices.

The recursion depth is too much for Windows' default stack size of 1 MB, so increasing it with the help of compiler flags is a good idea. The recursion depth can be significantly reduced by removing the ingredients that everyone likes(by 2863 for the biggest dataset).

If the simulation has not fully finished within a 30 minute timeframe, it is forcefully stopped and the best solution found so far is chosen.

Finally, **local search** is used on the simulation solution, which leads to a roughly 10% score improvement.

### Scoring

| Input File     | Score  |
|----------------|--------|
| a_an_example   | 2      |
| b_basic        | 5      |
| c_coarse       | 5      |
| d_difficult    | 1,780  |
| e_elaborate    | 1,709  |
| **Total**      | 3,501  |

## Solution 2

### Strategy

This hybrid approach builds on the previous: an initial solution is produced by the simulation(with the timer reduced from 30 minutes to only 2 minutes). Then, a **bounded priority queue** with capacity 10 is initialized from 10 runs of **simulated annealing**, followed by local search.

Finally, within a 25 minute timeframe, multiple batches of simulated annealing are repeatedly performed(10 in parallel at a time). The starting states for the simulated annealing are taken from the bounded priority queue(**multistart** metaheuristic), which contains the best 10 solutions found so far. This further helps with getting out of local minima. Whenever a better solution is found, it is pushed to the priority queue, which discards the previous 10th scoring solution.

### Scoring

| Input File     | Score  |
|----------------|--------|
| a_an_example   | 2      |
| b_basic        | 5      |
| c_coarse       | 5      |
| d_difficult    | 1,805  |
| e_elaborate    | 1,725  |
| **Total**      | 3,542  |