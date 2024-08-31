import math
import numpy as np

from Assets.Agent import Agent
from Assets.Game import Game
import plotly.graph_objects as go
import plotly.figure_factory as ff

class Graph:
    def  __init__(self):
        pass

    @staticmethod
    def render_epsilon_greedy_plot(game: Game, Agents_list: list[Agent],in_percentage:bool = True, all_Agents:bool = False):
        x = np.arange(Agents_list[0].get_total_N())
        tmp = None
        if not in_percentage:
            tmp = np.array([np.array(el) for el in Agents_list[0].get_mean_rewards_evolution()])
        else:
            tmp = np.array([np.array(el) / sum(el) if np.linalg.norm(el) != 0 else [1/2]*len(game.legal_actions()) for el in Agents_list[0].get_mean_rewards_evolution()])
        y1 = tmp[:,0]
        y2 = tmp[:,1]

        fig = go.Figure(data=[go.Scatter(x=x, y=y1, mode='lines', name='x1_first_Agent')])
        fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='x2_first_Agent'))

        if game.get_game_name() == "rock_paper_scissors":
            y3 = np.array([el[2] for el in Agents_list[0].get_mean_rewards_evolution()])
            fig.add_trace(go.Scatter(x=x, y=y3, mode='lines', name='x3_first_Agent'))

        if all_Agents:
            y4 = np.array([el[0] for el in Agents_list[1].get_mean_rewards_evolution()])
            y5 = np.array([el[1] for el in Agents_list[1].get_mean_rewards_evolution()])
            fig.add_trace(go.Scatter(x=x, y=y4, mode='lines', name='x1_second_Agent'))
            fig.add_trace(go.Scatter(x=x, y=y5, mode='lines', name='x2_second_Agent'))
            if game.get_game_name() == "rock_paper_scissors":
                y6 = np.array([el[2] for el in Agents_list[1].get_mean_rewards_evolution()])
                fig.add_trace(go.Scatter(x=x, y=y6, mode='lines', name='x3_second_Agent'))

        fig.update_layout(title='Vector Field and Multiple Graphs',
                        xaxis_title='X Axis',
                        yaxis_title='Y Axis',
                        legend_title='Graphs')
        
        fig.show()
    
    @staticmethod
    def compute_expected_maximum_payoff(matrix: np.array, vector: np.array, kappa:int) -> np.array:
        dim = matrix.shape[0]
        result = np.zeros(dim)
        for i in range(dim):
            for j in range(dim):
                if not np.isclose(vector[j], 0):
                    result[i] += matrix[i,j]*vector[j] *(np.sum(vector[matrix[i,:] <= matrix[i,j]])**kappa - np.sum(vector[matrix[i,:] < matrix[i,j]])**kappa) / np.sum(vector[matrix[i,:] == matrix[i,j]])
        return result

    @staticmethod
    def compute_vector_field(game: Game, alpha, tau, kappa, normalise:bool = True):
        A, B = game.get_bi_matrix_seperated()
        N = 20
        x, y = np.meshgrid(np.linspace(0, 1, N + 2)[1:-1], np.linspace(0, 1, N +2)[1:-1], indexing= 'xy')

        dx = np.zeros_like(x)
        dy = np.zeros_like(y)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if len(game.legal_actions()) == 3 and x[j,i] + y[j,i] >= 1:
                    continue
                if len(game.legal_actions()) == 2:
                    freq_dist_y = np.array([y[j,i], 1-y[j,i]]).T
                    freq_dist_x = np.array([x[j,i], 1-x[j,i]]).T
                    freq_index_x_axis = freq_index_y_axis = 0
                elif len(game.legal_actions()) == 3: 
                    # same vector as this vector has dependcies on both axis
                    #only valid when self-play --> same reward matrix
                    freq_dist_y = freq_dist_x = np.array([x[j,i], y[j,i], 1-x[j,i] - y[j,i]]).T
                    freq_index_x_axis = 0
                    freq_index_y_axis = 1
                else:
                    raise Exception("Graphs only defined for two and three legal actions")
                u = Graph.compute_expected_maximum_payoff(A,freq_dist_y,kappa) #has to be distribution other Agent as this prob determine the reward of this Agent when action is chosen (action is chosen as index in vector)
                w = Graph.compute_expected_maximum_payoff(B,freq_dist_x,kappa)
                factor_1_1 = alpha / tau * (u[freq_index_x_axis] - freq_dist_x.T@u)
                factor_2_1 = alpha / tau * (w[freq_index_y_axis] - freq_dist_y.T@w)
                tmp_1_2 = 0
                tmp_2_2 = 0
                if not np.isclose(freq_dist_x[freq_index_x_axis], 0):
                    for q in range(len(game.legal_actions())):
                        if not np.isclose(freq_dist_x[q], 0):
                            tmp_1_2 += freq_dist_x[q] * math.log10(freq_dist_x[q]/freq_dist_x[freq_index_x_axis])
                            tmp_2_2 += freq_dist_y[q] * math.log10(freq_dist_y[q]/freq_dist_y[freq_index_y_axis])
                factor_1_2 = alpha * tmp_1_2
                factor_2_2 = alpha * tmp_2_2
                tmp_x = (factor_1_1 + factor_1_2)*x[j,i]
                tmp_y = (factor_2_1 + factor_2_2)*y[j,i]
                if normalise:
                    norm = np.linalg.norm([tmp_x,tmp_y])
                    tmp_x = tmp_x/norm if not np.isclose(norm, 0) else tmp_x
                    tmp_y = tmp_y / norm if not np.isclose(norm, 0) else tmp_y
                dx[j,i] = tmp_x
                dy[j,i] = tmp_y

        fig = ff.create_quiver(x,y,dx, dy,
                                name=f"Vector plot",
                                line_width = 1,
                                scale= .025,
                                arrow_scale=0.2
                                )
        return fig

    @staticmethod
    def render_lenient_plot(game: Game, Agents_list: list[Agent], alpha, tau, kappa , normalise:bool = True):
        
        fig = Graph.compute_vector_field(game, alpha, tau , kappa)

        if len(game.legal_actions()) == 2:
            x_trace1 = np.array([freq0[0] for freq0 in Agents_list[0].get_distributions_evolution()])
            y_trace1 = np.array([freq0[0] for freq0 in Agents_list[1].get_distributions_evolution()])
        else: #exception will already been raised
            x_trace1 = np.array([freq0[0] for freq0 in Agents_list[0].get_distributions_evolution()])
            y_trace1 = np.array([freq0[1] for freq0 in Agents_list[0].get_distributions_evolution()])
        

        fig.add_trace(go.Scatter(x=x_trace1, y=y_trace1, mode='lines', name=f"Distribution [{float(x_trace1[0]), float(y_trace1[0])}]"))
        fig.add_trace(go.Scatter(x=[x_trace1[0], x_trace1[-1]], y=[y_trace1[0],y_trace1[-1]],
                    mode='markers',
                    marker_size=8,
                    marker = {
                        "color":"black"
                    },
                    text = ["begin" , "end"],
                    showlegend = False))
        if len(game.legal_actions()) == 2:
            fig.update_layout(title='Vector Field and Multiple Graphs',
                            xaxis_title='X Axis',
                            yaxis_title='Y Axis',
                            legend_title='Graphs')
        else:
            fig.update_layout(title='Vector Field and Multiple Graphs',
                            xaxis_title='X1 Axis',
                            yaxis_title='X2 Axis',
                            legend_title='Graphs')
        # Show the plot
        fig.show()

    @staticmethod
    def render_2_actions_graph(game: Game, Agents_list: list[Agent], alpha, tau, normalise:bool = True):
        fig = Graph.compute_vector_field(game, alpha, tau , kappa=1)

        x_trace1 = np.array([freq0[0] for freq0 in Agents_list[0].get_distributions_evolution()])
        y_trace1 = np.array([freq0[0] for freq0 in Agents_list[1].get_distributions_evolution()])


        fig.add_trace(go.Scatter(x=x_trace1, y=y_trace1, mode='lines', name=f"Distribution [{float(x_trace1[0]), float(y_trace1[0])}]"))
        fig.add_trace(go.Scatter(x=[x_trace1[0], x_trace1[-1]], y=[y_trace1[0],y_trace1[-1]],
                    mode='markers',
                    marker_size=8,
                    marker = {
                        "color":"black"
                    },
                    text = ["begin" , "end"],
                    showlegend = False))
        fig.update_layout(title='Vector Field and Multiple Graphs',
                        xaxis_title='X Axis',
                        yaxis_title='Y Axis',
                        legend_title='Graphs')

        # Show the plot
        fig.show()

    @staticmethod
    def render_3_actions_graph(game, Agents_list, normalise: bool = True):

        A, _ = game.get_bi_matrix_seperated()

        N = 10
        # We plot x1 and x2 of the same Agents instead of X1 of both Agents
        # We can only do this because we used self-play --> same Agents, same reward matrix
        x1, x2 = np.meshgrid(np.linspace(0, 1, N + 2)[1:-1], np.linspace(0, 1, N + 2)[1:-1], indexing="ij")
        # formula = lambda x: np.array([x, 1-x]).T
        # vector_agent_1 = formula(x)
        # vector_agent_2 = formula(y)

        dx = np.empty_like(x1)
        dy = np.empty_like(x1)
        for i in range(x1.shape[0]):
            for j in range(x1.shape[1]):
                if x1[j,i] + x2[j,i] >= 1:
                    continue
                #There is only vector as this vector is dependent on both axis
                tmp_vector_x = np.array([x1[j,i],x2[j,i], 1 - x1[j,i] - x2[j,i]]).T
                tmp_x = x1[j,i]*((A@tmp_vector_x)[0] - tmp_vector_x.T@A@tmp_vector_x)
                tmp_y = x2[j,i]*((A@tmp_vector_x)[0] - tmp_vector_x.T@A@tmp_vector_x)
                if normalise:
                    norm = np.linalg.norm([tmp_x,tmp_y])
                    tmp_x = tmp_x/norm if not np.isclose(norm, 0) else tmp_x
                    tmp_y = tmp_y/ norm if not np.isclose(norm, 0) else tmp_y
                dx[j,i] = tmp_x
                dy[j,i] = tmp_y

        
        fig = ff.create_quiver(x1,x2,dx, dy,
                                name='Q-learning Vector plot',
                                line_width = 1,
                                scale= .025,
                                arrow_scale=0.2
                                )

        x_trace1 = np.array([freq0[0] for freq0 in Agents_list[0].get_distributions_evolution()])
        y_trace1 = np.array([freq0[1] for freq0 in Agents_list[0].get_distributions_evolution()])

        fig.add_trace(go.Scatter(x=x_trace1, y=y_trace1, mode='lines', name=f"Distribution [{float(x_trace1[0]), float(y_trace1[0])}]"))

        fig.update_layout(title='Vector Field and Multiple Graphs',
                        xaxis_title='X1 Axis',
                        yaxis_title='X2 Axis',
                        legend_title='Graphs')

        # Show the plot
        fig.show()

    @staticmethod
    def render_combined_2_actions_graph(game: Game, all_traces: list[list[list[list]]], initial_Q_values_list, alpha, gamma, tau, learning_name, kappa = 1, normalise: bool = True):
        # kappa = 1 -> Q learning
        fig = Graph.compute_vector_field(game, alpha, tau , kappa=kappa)

        colors = ['red', 'blue', 'green', 'yellow']

        for idx, traces in enumerate(all_traces):
            x_trace1 = np.array([freq0[0] for freq0 in traces[0]])
            y_trace1 = np.array([freq0[0] for freq0 in traces[1]])

            label = f"Trace {idx+1}: P1={x_trace1[0]:.2f}, P2={y_trace1[0]:.2f}"
            fig.add_trace(go.Scatter(x=x_trace1, y=y_trace1, mode='lines', line=dict(color=colors[idx]), name=label))

            fig.add_trace(go.Scatter(
            x=[x_trace1[0]], 
            y=[y_trace1[0]], 
            mode='markers', 
            marker=dict(size=8, color='green'), 
            name=f"Start {label}",
            showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=[x_trace1[-1]], 
                y=[y_trace1[-1]], 
                mode='markers', 
                marker=dict(size=8, color='red'),  
                name=f"End {label}",
                showlegend=False 
            ))

        title = f"{game.get_game_name()} | {learning_name.replace('_', ' ')}\n"
        title += f"| alpha={alpha}, gamma={gamma}, tau={tau}, kappa={kappa}"

        fig.update_layout(title=title,
                        xaxis_title='Player 1 Action 0 Probability',
                        yaxis_title='Player 2 Action 0 Probability',
                        legend_title='Traces',
                        xaxis=dict(range=[0, 1]),  # Set X axis range between 0 and 1
                        yaxis=dict(range=[0, 1]))   # Set Y axis range between 0 and 1)

        # Show the plot
        fig.show()

    @staticmethod
    def render_combined_2_actions_graph_FAQ(game: Game, all_traces: list[list[list[list]]], initial_Q, alpha, gamma, tau, learning_names, kappa = 1, normalise: bool = True):
        # kappa = 1 -> Q learning
        fig = Graph.compute_vector_field(game, alpha, tau , kappa=kappa)

        colors = ['red', 'blue', 'green', 'yellow']

        for idx, traces in enumerate(all_traces):
            x_trace1 = np.array([freq0[0] for freq0 in traces[0]])
            y_trace1 = np.array([freq0[0] for freq0 in traces[1]])

            label = f"Trace {idx+1}: P1={x_trace1[0]:.2f}, P2={y_trace1[0]:.2f}"
            fig.add_trace(go.Scatter(x=x_trace1, y=y_trace1, mode='lines', line=dict(color=colors[idx]), name=label))

            fig.add_trace(go.Scatter(
            x=[x_trace1[0]], 
            y=[y_trace1[0]], 
            mode='markers', 
            marker=dict(size=8, color='green'), 
            name=f"Start {label}",
            showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=[x_trace1[-1]], 
                y=[y_trace1[-1]], 
                mode='markers', 
                marker=dict(size=8, color='red'),  
                name=f"End {label}",
                showlegend=False 
            ))

        title = f"{game.get_game_name()} | {learning_names[0].replace('_', ' ')} | {learning_names[1].replace('_', ' ')}\n"
        title += f"| alpha={alpha}, gamma={gamma}, tau={tau}, kappa={kappa}"

        fig.update_layout(title=title,
                        xaxis_title='Player 1 Action 0 Probability',
                        yaxis_title='Player 2 Action 0 Probability',
                        legend_title='Traces',
                        xaxis=dict(range=[0, 1]),  # Set X axis range between 0 and 1
                        yaxis=dict(range=[0, 1]))   # Set Y axis range between 0 and 1)

        # Show the plot
        fig.show()

    