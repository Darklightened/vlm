import torch
import torchsort

x_ = [0.0011 for i in range(2300)]

x = torch.tensor([x_], requires_grad=True).cuda()
x = x.to(torch.float16)
y = torchsort.soft_rank(x, regularization_strength=1.0)

print(y)

print(torch.autograd.grad(y[0, 0], x))