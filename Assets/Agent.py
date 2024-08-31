from copy import deepcopy
import math
from Assets.Game import Game
import numpy as np


class Agent():
  __learning_string = str()
  __game: Game
  #Q-learning
  __Q = None
  __alpha = None
  __beta = None
  __gamma = None
  __tau = None
  __temperature_reduction_function = None
  __index_latest_action = -1
  __previous_distributions  = None
  __execution_count:int = 0
  #additional lenient_Q_learning
  __kappa:str  = None
  __lenience_reduction_function = None
  __previous_rewards: list[list[int]]= None

  #epsilon_greedy
  __epsilon = None
  __mean_reward_per_action = None #list
  __N: list = None


  def __init__(self,game: Game, learning_string:str,epsilon:float = None, initial_Q:list = None, alpha:float = None, beta:float = None, gamma:float = None, tau:float = None, temperature_reduction_function = None, kappa: int = None, lenience_reduction_function = None):
    self.__game = game
    self.__learning_string = learning_string

    if "Q_learning" in learning_string:
      assert(initial_Q is not None)
      assert(alpha  is not None)
      assert(gamma  is not None)
      assert(tau    is not None)  
      self.__previous_distributions = list()
      initial_distribution = list()
      for prob_index in range(len(game.legal_actions())):
        initial_distribution.append((np.exp(initial_Q[prob_index] / tau)) / sum([np.exp(Q_value / tau) for  Q_value in initial_Q]))
      self.__previous_distributions.append(initial_distribution)

      self.__Q = deepcopy(initial_Q)
      self.__alpha = alpha 
      self.__gamma = gamma
      self.__tau = tau
      if "frequency_adjusted" not in learning_string:
        self.__alpha = alpha
        self.__beta = 1
        self.__temperature_reduction_function = temperature_reduction_function if temperature_reduction_function is not None else lambda count,tau: tau
      else:
        assert(beta is not None)
        # tmp_tau, tmp_beta = self.__compute_optimal_parameters_for_FAQL(game, beta, tau, gamma)
        # tmp_tau_list = [tau, tau, tmp_tau]
        # tmp_beta_list = [beta, tmp_beta, beta]
        # print(f"1. keep given values tau = {tau} and beta = {beta}")
        # print(f"2. compute optimal beta = {tmp_beta} with tau = {tau}")
        # print(f"3. compute optimal tau = {tmp_tau} with beta = {beta}")
        # index = int(input("Give your choice: "))
        # self.__tau = tmp_tau_list[index-1]
        # self.__beta = tmp_beta_list[index-1]
        self.__tau = tau
        self.__beta = beta
      if "lenient" in learning_string:
        assert(kappa is not None)
        self.__kappa = kappa if kappa >=1 else 1
        self.__lenience_reduction_function = lenience_reduction_function if lenience_reduction_function is not None else lambda count,kappa: kappa
        self.__previous_rewards = [list() for _ in range(len(game.legal_actions()))]
    elif learning_string == "epsilon_greedy":
      assert(epsilon is not None)
      self.__epsilon = epsilon
      self.__mean_reward_per_action = list()
      self.__mean_reward_per_action.append([0]*len(game.legal_actions()))
      self.__N = [0]*len(game.legal_actions())
  
  @staticmethod
  def __compute_optimal_parameters_for_FAQL(game: Game, original_beta, original_tau, gamma):
    min_reward, max_reward = game.get_extrema_rewards()
    min_reward*= 1/(1-gamma); max_reward *= 1/(1-gamma)
    pseudo_Q_list = [min_reward]
    while len(pseudo_Q_list) < len(game.legal_actions()):
      pseudo_Q_list.append(max_reward)
    beta = 0.9 * (np.exp(pseudo_Q_list[0] / original_tau)) / sum([np.exp(Q_value / original_tau) for  Q_value in pseudo_Q_list])
    tau = 1.1 * pseudo_Q_list[0] / (math.log10(1/original_beta) - math.log10(sum([np.exp(Q_value / original_tau) for  Q_value in pseudo_Q_list])))
    return beta, tau
  
  @staticmethod
  def __normalize(input_list):
    total = sum(input_list)
    return[el/total for el in input_list]
  
  def get_action(self):
    if "Q_learning" in self.__learning_string:
      return self.__get_action_Q_learning()
    elif self.__learning_string == "epsilon_greedy":
      return self.__get_action_epsilon_greedy_learning()
    else:
       raise Exception("Learning algorithm not implemented")

  def __get_action_Q_learning(self) -> int:
    #action = random.choice(state.legal_actions(state.current_Agent()))
    self.__index_latest_action = np.random.choice(range(len(self.__game.legal_actions())), p = self.__previous_distributions[-1])
    return self.__game.legal_actions()[self.__index_latest_action]
  
  def __get_action_epsilon_greedy_learning(self) -> int:
    p = np.random.random()
    if p < self.__epsilon:
      self.__index_latest_action = np.random.choice(range(len(self.__game.legal_actions())))
    else:
      self.__index_latest_action = np.argmax(self.__mean_reward_per_action[-1])
    return self.__game.legal_actions()[self.__index_latest_action]

  def learn(self, reward):
    self.__tau = self.__temperature_reduction_function(self.__execution_count, self.__tau) if self.__temperature_reduction_function is not None else self.__tau
    
    if "Q_learning" in self.__learning_string:
      if "lenient" in self.__learning_string:
        self.__lenient_Q_learning(reward)
      else:
        self.__frequency_adjusted_Q_learning(reward)
    elif self.__learning_string == "epsilon_greedy":
      self.__epsilon_greedy_learning(reward)
    else:
      raise Exception("Learning algorithm not implemented")
    
    self.__execution_count +=1

  def __lenient_Q_learning(self,reward):
    self.__previous_rewards[self.__index_latest_action].append(reward)
    new_kappa = self.__lenience_reduction_function(self.__execution_count, self.__kappa)
    if self.__kappa != new_kappa : #kappa changed
      self.__kappa = new_kappa if new_kappa >= 1 else 1
      for index in range(len(self.__previous_rewards)):
        self.__index_latest_action = index
        if len(self.__previous_rewards[self.__index_latest_action]) >= self.__kappa: 
          actual_reward = max(self.__previous_rewards[self.__index_latest_action])
          self.__previous_rewards[self.__index_latest_action] = list()
          self.__frequency_adjusted_Q_learning(actual_reward)
      self.__index_latest_action = -1
    elif len(self.__previous_rewards[self.__index_latest_action]) >= self.__kappa: #should never be more, just equal
      actual_reward = max(self.__previous_rewards[self.__index_latest_action])
      self.__previous_rewards[self.__index_latest_action] = list()
      self.__frequency_adjusted_Q_learning(actual_reward)
    else:
      self.__previous_distributions.append(deepcopy(self.__previous_distributions[-1]))
      self.__index_latest_action = -1
  
  def __epsilon_greedy_learning(self, reward):
    self.__N[self.__index_latest_action] +=1
    self.__mean_reward_per_action.append(deepcopy(self.__mean_reward_per_action[-1]))
    self.__mean_reward_per_action[-1][self.__index_latest_action] = (1 - 1/self.__N[self.__index_latest_action]) * self.__mean_reward_per_action[-1][self.__index_latest_action] + reward / self.__N[self.__index_latest_action]

  def __frequency_adjusted_Q_learning(self, reward) -> None:
    assert(self.__index_latest_action >= 0)
    self.__Q[self.__index_latest_action] = self.__Q[self.__index_latest_action] + min(self.__beta/self.__previous_distributions[-1][self.__index_latest_action],1) * self.__alpha * (reward + self.__gamma * max(self.__Q) - self.__Q[self.__index_latest_action])
    #policy updating
    distribution = deepcopy(self.__previous_distributions[-1])
    for prob_index in range(len(self.__previous_distributions[-1])):
      distribution[prob_index] = (np.exp(self.__Q[prob_index] / self.__tau)) / sum([np.exp(Q_value / self.__tau) for  Q_value in self.__Q])
    self.__previous_distributions.append(deepcopy(distribution))
    self.__index_latest_action = -1

  # def done_learning(self) -> bool:
  #   if len(self.__previous_distributions) < 2 : return False

  #   distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(self.__previous_distributions[-2], self.__previous_distributions[-1])))
  #   return distance <= self.__learning_threshold
  
  def latest_error(self)-> float:
    if len(self.__previous_distributions) < 2: return math.nan

    return math.sqrt(sum((a - b) ** 2 for a, b in zip(self.__previous_distributions[-2], self.__previous_distributions[-1])))
    
  def get_Q(self) -> list:
    return deepcopy(self.__Q)
  
  def get_distribution(self) -> list:
    return deepcopy(self.__previous_distributions[-1])
  
  def get_distributions_evolution(self) -> list[list]:
    return deepcopy(self.__previous_distributions)
  
  def get_mean_reward(self) -> list:
    return deepcopy(self.__mean_reward_per_action[-1])
  
  def get_mean_rewards_evolution(self) -> list[list]:
    return deepcopy(self.__mean_reward_per_action)
  
  def get_N(self) -> list:
    return deepcopy(self.__N)
  
  def get_total_N(self) -> int:
    return sum(self.get_N())
