

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
                                name=f"Vector plot \n  alpha={alpha}, tau={tau}, kappa={kappa}",
                                line_width = 1,
                                scale= .025,
                                arrow_scale=0.2
                                )
        return fig