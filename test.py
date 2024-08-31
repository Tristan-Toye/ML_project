import numpy as np

tau = 0.1
Q = [0.78 , 0.22]
distribution = [0,0]
for prob_index in range(2):
    distribution[prob_index] = (np.exp(Q[prob_index]) / tau) / sum([np.exp(Q_value) / tau for  Q_value in Q])

print(distribution)