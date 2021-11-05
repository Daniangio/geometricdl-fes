import torch
import json
import numpy as np

FILE = 'data/ala_dipep/phi-psi-free-energy.txt'

with open(FILE, "r") as t:
    tensor = torch.as_tensor([[float(phi), float(psi), float(f)] for _, phi, psi, f in [x.split() for index, x in enumerate(t.readlines()) if index < 10000]])
    phi_t = tensor[:, 0]
    psi_t = tensor[:, 1]

left, right = [], []

for i, (phi, psi) in enumerate(zip(phi_t, psi_t)):
    if phi < -np.pi/4 and psi > -np.pi/3:
        left.append(i)
    
    if phi > np.pi/8 and phi < np.pi/2 and psi > -np.pi*7/8 and psi < np.pi/4:
        right.append(i)

with open('left.json', 'w') as f:
    json.dump(left, f)
with open('right.json', 'w') as f:
    json.dump(right, f)