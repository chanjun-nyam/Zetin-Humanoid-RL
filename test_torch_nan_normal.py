import torch as th

device = th.device('cuda:0')

a = th.tensor([6, 1, float('nan')], device=device)
b = th.tensor([3, 2, 8], device=device)

x = th.normal(a, b)
print(x)

x = th.normal(b, a)
print(x)
