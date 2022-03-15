from strategy import Strategy
from T4exps import experiment_execution, best_strategy, terminate, t4e_print

## The experimental design
def parameter_tuning(S, param, param_values):
    original_value = S.params[param]
    original_name = S.name
    params = S.params.copy()
    for value in param_values:
        params[param]=value
        if original_value == value: 
            continue
        else: 
            S2 = Strategy.create_strategy(original_name, S.exe, S.params_str, params)
        S = best_strategy(S, S2)
    return S

def experiment():
    S = Strategy.create_strategy('BSG_CLP', 'bsg_algo/BSG_CLP', '--alpha={a} --beta={b} --gamma={g} -p {p} -t 1', {"a": 0.0, "b": 0.0, "g": 0.0, "p": 0.0})
    S = parameter_tuning(S, "a", [0.0, 1.0, 2.0, 4.0, 8.0])
    t4e_print(str(S.params["a"]) + " ")
    S = parameter_tuning(S, "b", [0.0, 0.5, 1.0, 2.0, 4.0])
    t4e_print(str(S.params["b"]) + " ")
    S = parameter_tuning(S, "g", [0.0, 0.1, 0.2, 0.3, 0.4])
    t4e_print(str(S.params["g"]) + " ")
    S = parameter_tuning(S, "p", [0.00, 0.01, 0.02, 0.03, 0.04])
    t4e_print(str(S.params["p"]) + " ")
    terminate()  


experiment_execution(experiment, 'bsg_algo/instancesCLP-shuf.txt', n_runs=4)


