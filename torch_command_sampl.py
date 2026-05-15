import torch as th

device = th.device('cpu')
device = th.device('cuda')

x = th.tensor([0, 1, 4, 2, 6, 3, 2], dtype=th.float32, device=device)
idx = [0, 1, 2]
idx = [0, 1, 0]

print(x)

x[idx] = th.tensor([991, 992, 993], dtype=th.float32, device=device)

print(x)