# T4exps:  A tool  for experimental algorithmics

A tool for assisting the algorithm developer (AD) in the experimental phase. 

The tool offers the AD some simple and intuitive commands and methods for designing a set of experiments, mainly based on comparing pairs of strategies.

When the designed experiment is executed, the goal of the tool is to report a probable output with a high degree of certainty in a fraction of the time required to execute the whole experiment. 

The tool achieves this by executing the experiment in an **incremental and speculative way**, i.e., as the experiment is executed, the winner of each comparison is guessed based on the partial results obtained so far.
In any moment, if the AD requires, the tool may report a probable output with a degree of certainty. As the execution continues, the output may change, but the degree of certainty increases. At the end, the output is 100\% certain, i.e., it is the same as if the experiments were executed in a sequential way. 


## Example of an experiment algorithm

The following code shows an example of an experiment algorithm using **T4exps**. 
In the code, `BSG_CLP` is the name of the executable of the tested algorithm. ``--alpha={a} --beta={b} --gamma={g} -p {p} -t 1`` is the string of arguments of the executable (terms in curly brackets in the argument string will be replaced by the corresponding instance or parameter value when the experiments are executed). `params` represents the parameter vector, storing the value of each parameter of the algorithm. The file `bsg_algo/instancesCLP-shuf.txt` contains the paths to the set of instances for the experiment.

````python
from strategy import Strategy
from T4exps import experiment_execution, best_strategy, terminate, t4e_print

def parameter_tuning(S, param, param_values):
    initial_value = S.params[param]
    name = S.name
    params = S.params.copy()
    for value in param_values:
        params[param]=value
        if initial_value == value: continue
        else: 
            S2 = Strategy.create_strategy(name, S.exe, S.params_str, params)
        S = best_strategy(S, S2)
    return S

def experiment():
    S = Strategy.create_strategy('BSG_CLP', 'bsg_algo/BSG_CLP', \
        '--alpha={a} --beta={b} --gamma={g} -p {p} -t 1', \
        params={"a": 0.0, "b": 0.0, "g": 0.0, "p": 0.0})

    S = parameter_tuning(S, "a", [0.0, 1.0, 2.0, 4.0, 8.0])
    S = parameter_tuning(S, "b", [0.0, 0.5, 1.0, 2.0, 4.0])
    S = parameter_tuning(S, "g", [0.0, 0.1, 0.2, 0.3, 0.4])
    S = parameter_tuning(S, "p", [0.00, 0.01, 0.02, 0.03, 0.04])

    t4e_print(str(S.params) + " ")

    terminate()  


experiment_execution(experiment, 'bsg_algo/instancesCLP-shuf.txt', n_runs=4)
````

The function `parameter_tuning` returns the best strategy found after testing the different values for the given parameter.

The experiment algorithm basically performs a set of comparisons for the algorithm `BSG_CLP` which has 4 parameters (`a`,`b`,`g` and `p`). The value of each parameter is tuned sequentially by comparing different alternatives. At the end, the code print on the screen a string describing the best configuration found by the experiment algorithm. A similar experiment was performed for analyzing a heuristic evaluation function for the BSG algorithm in [[1]](https://www.sciencedirect.com/science/article/pii/S0305054817300023).


## Running the experiment (incremental execution)

For running an experiment you can execute
````
python experiment.py
````

Note that for running the example you require to execute the file `BSG_CLP` compiled in linux. 

If you cannot run the example on your PC, you can test it on these [colab notebook](https://colab.research.google.com/drive/1ILbTdImp-_laOyJ4Y10D3cxUzUNbZNo4?usp=sharing).
