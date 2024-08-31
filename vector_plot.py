


from Assets.Game import Game
from Assets.Graph import Graph

legal_games = ["rock_paper_scissors", "subsidy_game", "Battle_of_the_sexes", "prisoners_dilemma"]
game_name = "subsidy_game"
alpha = 5*10**(-8)
tau = 2
game = Game(game_name)
kappa = 1

fig = Graph.compute_vector_field(game, alpha, tau , kappa=kappa, normalise= True)

title = f"{game.get_game_name()}\n"
title += f"| alpha={alpha}, tau={tau}, kappa={kappa}"

fig.update_layout(title=title,
                xaxis_title='Player 1 Action 0 Probability',
                yaxis_title='Player 2 Action 0 Probability',
                legend_title='Traces',
                xaxis=dict(range=[0, 1]),  # Set X axis range between 0 and 1
                yaxis=dict(range=[0, 1]))   # Set Y axis range between 0 and 1)

# Show the plot
fig.show()