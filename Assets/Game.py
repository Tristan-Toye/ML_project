from copy import deepcopy

import numpy as np


class Game():

  __game_name:str = str()
  __bi_matrix: list[list[list[int]]] = list()
  __legal_moves: list[int] = list()
  __rock_paper_scissors_bi_matrix = [[[0,0], [-0.05, 0.05], [0.25, -0.25]],
                                     [[0.05,-0.05], [0,0], [-0.5, 0.5]],
                                     [[-0.25, 0.25], [0.5, -0.5], [0,0]]
                                     ]

  __subsidy_game_bi_matrix = [[[12,12],[0,11]],
                              [[11,0],[10,10]]]
  
  __battle_of_the_sexes_bi_matrix = [[[3,2],[0,0]],
                                     [[0,0],[2,3]]]
  
  __prisoners_dilemma_bi_matrix = [[[-1,-1], [-4, 0]],
                                   [[0, -4], [-3, -3]]]
  
  def __init__(self, game_string: str) -> None:
    self.__game_name = game_string
    if game_string == "rock_paper_scissors":
      self.__bi_matrix = self.__rock_paper_scissors_bi_matrix
      self.__legal_moves = [0,1,2]
    elif game_string == "subsidy_game":
      self.__bi_matrix = self.__subsidy_game_bi_matrix
      self.__legal_moves = [0,1]
    elif game_string == "Battle_of_the_sexes":
      self.__bi_matrix = self.__battle_of_the_sexes_bi_matrix
      self.__legal_moves = [0,1]
    elif game_string ==  "prisoners_dilemma":
      self.__bi_matrix = self.__prisoners_dilemma_bi_matrix
      self.__legal_moves = [0,1]

  def get_extrema_rewards(self):
    return min([min(value) for row in self.__bi_matrix for value in row]), \
            max([max(value) for row in self.__bi_matrix for value in row])
  
  def get_reward(self, indexes: list) -> list[int]:
    return deepcopy(self.__bi_matrix[indexes[0]][indexes[1]])

  def legal_actions(self) -> list:
    return deepcopy(self.__legal_moves)
  
  def num_Agents(self) -> int:
    return 2
  
  def get_bi_matrix(self)->list[list[list[int]]]:
    return deepcopy(self.__bi_matrix)
  
  def get_bi_matrix_seperated(self) -> np.array:
    tmp = self.get_bi_matrix()
    return np.array([[a[0] for a in b] for b in tmp]), np.array([[a[1] for a in b] for b in tmp]).T
    
  def get_game_name(self):
    return self.__game_name
  