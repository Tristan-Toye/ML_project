
from Assets.Graph import Graph
from Assets.Agent import Agent
from Assets.Game import Game
import numpy as np

#line plots for RPS

def main():
  legal_games = ["rock_paper_scissors", "subsidy_game", "Battle_of_the_sexes", "prisoners_dilemma"]
  legal_learning_algos = ["espilon_greedy", "Q_learning", "lenient_Q_learning", "frequency_adjusted_Q_learning", "lenient_frequency_adjusted_Q_learning"]
  

  max_game_iterations:int = [100000, 5000000]
  game_name = "subsidy_game"
  learning_names = ["lenient_Q_learning","lenient_frequency_adjusted_Q_learning"]
  graph = Graph()

  # epsilon_greedy
  # First conclusion: smaller exploration finds pareto optimal (12,12) on subsidy, but higher exploration finds NE[self.__index_latest_action]
  epsilon = 0.4

  # Q-learning
  # Conclusion: for both Q-learning and Lenient Q_learning holds true that the model dynamics (dx , dy) in distribution (x , y) are linear dependent on the value of their respective distribution 
  # aka dx is linear dependent on x only, dx is linear dependent on y only (assumin static model ofc)
  initial_Q = [[0.25, 0.5],[0.25, 0.25]] #also initial prob distribution (list gets normalised auto for this, not for Q)

  alpha = 5*10**(-5)
  gamma = 0.9
  tau = 0.3

  def temperature_reduction_function(count:int, tau):
    return tau

  #lenient_Q_learning
  kappa = 10

  #if no reduction --> converts to 12/12 in subsidy (kappa  == 4)
  def lenience_reduction_function(count:int, kappa:int):
    return kappa
  
  #frequency_adjusted
  beta = 10**(-3)


  normalise_vector_plot = False
  all_traces = []

  for idx, learning_name in enumerate(learning_names):
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
                                    alpha=alpha*10**2,
                                    gamma=gamma,
                                    tau=tau,
                                    beta= beta
                                  )     
                            )
      elif learning_name == "lenient_frequency_adjusted_Q_learning":
        Agents_list.append(Agent(
                                    game = game,
                                    learning_string=learning_name,
                                    initial_Q=initial_Q[index],
                                    alpha=alpha*10**2,
                                    gamma=gamma,
                                    tau=tau,
                                    beta= beta,
                                    temperature_reduction_function=temperature_reduction_function,
                                    kappa=kappa,
                                    lenience_reduction_function=lenience_reduction_function
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
    while count < max_game_iterations[idx]:
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
      #graph.render_lenient_plot(game, Agents_list, alpha= alpha, tau = tau, kappa = kappa, normalise= normalise_vector_plot)
      graph.render_combined_2_actions_graph_FAQ(game, all_traces, initial_Q, alpha, gamma, tau, learning_names, kappa = kappa, normalise=normalise_vector_plot)
    else:
        graph.render_combined_2_actions_graph_FAQ(game, all_traces, initial_Q, alpha, gamma, tau, learning_names, normalise=normalise_vector_plot)


  
if __name__ == "__main__":
  main()


