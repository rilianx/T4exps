import subprocess
import multiprocessing
import copy

class Strategy:
    strategy_dict = dict()

    def get_id(self):
      return str(tuple(params.values())) + " " + exe + " " + name

    @staticmethod
    def create_strategy(name, exe, params_str, params):
      str_id = str(tuple(params.values())) + " " + exe + " " + name
      if str_id in Strategy.strategy_dict:
        return Strategy.strategy_dict[str_id]
      else:
        S = Strategy(name, exe, params_str, params)
        Strategy.strategy_dict[str_id]=S
        return S

    def __init__(self, name, exe, params_str, params):
      self.name=name
      self.exe=exe
      self.params_str=params_str
      self.params=params
      self.results=None
      self.est_means=None
      self.needs_to_be_sampled=False
      self.n_runs=0
      self.no_impact=1


    def run(self, instance, i, PI, ret_dic):
      commando = self.exe + " " + instance.rstrip() + " " + self.params_str.format(**self.params)
      output = subprocess.getoutput(commando)
      output = output.splitlines()
      ret_dic[i] = float(output[-1])  

    def run_instances(self):
      return self.n_runs

    def norm_results(self, base_strategy=None):
      n = self.run_instances()
      res = self.results[0:n]
      if base_strategy is not None:
        res -= base_strategy.results[0:n]
      return res
