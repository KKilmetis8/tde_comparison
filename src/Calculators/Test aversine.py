""" TEST HAVERSINE """

import numpy as np

obs_theta = np.pi /3
cell_theta = np.pi /5
obs_phi = np.pi /5
cell_phi = np.pi /8
delta_theta = obs_theta - cell_theta
delta_phi = obs_phi - cell_phi

# Haversine formula from other
arg = np.sin(delta_theta / 2)**2 + np.cos(obs_theta) * np.cos(cell_theta) * np.sin(delta_phi/2)**2
value_from_other = 2 * np.arctan2( np.sqrt(arg), np.sqrt(1-arg))
print(value_from_other)

# Haversine formula from us
our_value = 2 * np.arcsin(np.sqrt((np.sin(delta_theta/2))**2 + np.cos(cell_theta) * np.cos(obs_theta) * (np.sin(delta_phi/2))**2 ))
print(our_value)