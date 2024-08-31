
from Assets.Graph import Graph
from Assets.Agent import Agent
from Assets.Game import Game
import numpy as np

#line plots for RPS

def main():
  legal_games = ["rock_paper_scissors", "subsidy_game", "Battle_of_the_sexes", "prisoners_dilemma"]
  legal_learning_algos = ["espilon_greedy", "Q_learning", "lenient_Q_learning", "frequency_adjusted_Q_learning", "lenient_frequency_adjusted_Q_learning"]
  

  max_game_iterations:int = 10000
  game_name = "Battle_of_the_sexes"
  learning_name = "Q_learning"
  graph = Graph()

  # epsilon_greedy
  # First conclusion: smaller exploration finds pareto optimal (12,12) on subsidy, but higher exploration finds NE[self.__index_latest_action]
  epsilon = 0.4

  # Q-learning
  # Conclusion: for both Q-learning and Lenient Q_learning holds true that the model dynamics (dx , dy) in distribution (x , y) are linear dependent on the value of their respective distribution 
  # aka dx is linear dependent on x only, dx is linear dependent on y only (assumin static model ofc)
  initial_Q = [[0.8, 0.2],[0.6, 0.4]] #also initial prob distribution (list gets normalised auto for this, not for Q)
  initial_Q_values_list = [
        [[0.5, 0.5], [0.5, 0.5]],
        [[0.8, 0.2], [0.8, 0.2]],
        [[0.2, 0.8], [0.8, 0.2]],
        [[0.8, 0.2], [0.2, 0.8]]
    ]
  alpha = 5*10**(-4)
  gamma = 0.9
  tau = 0.1

  def temperature_reduction_function(count:int, tau):
    return tau

  #lenient_Q_learning
  kappa = 1

  #if no reduction --> converts to 12/12 in subsidy (kappa  == 4)
  def lenience_reduction_function(count:int, kappa:int):
    return kappa
  
  #frequency_adjusted
  beta = 1


  normalise_vector_plot = True
  all_traces = []

  for idx, initial_Q in enumerate(initial_Q_values_list):
    print(f"Running simulation {idx + 1} with initial Q-values: {initial_Q}")
    print("Creating game: " + game_name)
    game = Game(game_name)
    Agents_list: list[Agent] = list()
    assert(game.num_Agents() == len(initial_Q))
    for index in range(game.num_Agents()):
      if learning_name == "Q_learning":
        Agents_list.append(Agent(
                                    game = game,
                                    learning_string=learning_name,
                                    initial_Q=initial_Q[index],
                                    alpha=alpha,
                                    gamma=gamma,
                                    tau=tau,
                                    temperature_reduction_function=temperature_reduction_function
                                    )
                            )
      elif learning_name == "lenient_Q_learning":
        Agents_list.append(Agent(
                                    game = game,
                                    learning_string=learning_name,
                                    initial_Q=initial_Q[index],
                                    alpha=alpha,
                                    gamma=gamma,
                                    tau=tau,
                                    temperature_reduction_function=temperature_reduction_function,
                                    kappa=kappa,
                                    lenience_reduction_function=lenience_reduction_function
                                    )
                            )
      elif learning_name == "frequency_adjusted_Q_learning":
        Agents_list.append(Agent(
                                    game = game,
                                    learning_string=learning_name,
                                    initial_Q=initial_Q[index],
                                    alpha=alpha,
                                    gamma=gamma,
                                    tau=tau,
                                    beta= beta
                                  )     
                            )
      elif learning_name == "epsilon_greedy":
        Agents_list.append(Agent(
                                    game = game,
                                    learning_string=learning_name,
                                    epsilon= epsilon,
                                    
                                  )
                            )
      else: raise Exception("No such learning implemented")

    count = 0
    # and not all([Agent.done_learning() for Agent in Agents_list])
    while count < max_game_iterations:
      actions = [Agent.get_action() for Agent in Agents_list]
      rewards = game.get_reward(actions)
      [Agents_list[index].learn(reward) for index, reward in enumerate(rewards)]
      
      count+=1
      print(f"Played Games: {count}")

    traces = [agent.get_distributions_evolution() for agent in Agents_list]
    all_traces.append(traces)

  if "Q_learning" in learning_name:
    print("Agent 1:")
    print(Agents_list[0].get_distribution())
    print(Agents_list[0].get_Q())
    print(Agents_list[0].latest_error())
    print("Agent 2:")
    print(Agents_list[1].get_distribution())
    print(Agents_list[1].get_Q()) 
    print(Agents_list[1].latest_error())

    if "lenient" in learning_name:
      graph.render_lenient_plot(game, Agents_list, alpha= alpha, tau = tau, kappa = kappa, normalise= normalise_vector_plot)
    else:
      if game_name != "rock_paper_scissors":
        graph.render_combined_2_actions_graph(game, all_traces, initial_Q_values_list, alpha, gamma, tau, learning_name, normalise=normalise_vector_plot)
        #graph.render_combined_2_actions_graph(game, all_traces, normalise=normalise_vector_plot)
        #graph.render_2_actions_graph(game, Agents_list, normalise=normalise_vector_plot)
      else:
        graph.render_3_actions_graph(game, Agents_list, normalise = normalise_vector_plot)
  elif learning_name == "epsilon_greedy":
    print("Agent 1:")
    print(Agents_list[0].get_mean_reward())
    print(Agents_list[0].get_N())
    print("Agent 2:")
    print(Agents_list[1].get_mean_reward())
    print(Agents_list[1].get_N())

    graph.render_epsilon_greedy_plot(game,Agents_list)

  
if __name__ == "__main__":
  main()


