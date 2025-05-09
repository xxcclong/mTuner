import torch
import torch.optim as optim


class AFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        print(x, "forward")
        return x + 1

    @staticmethod
    def backward(ctx, grad_x):
        print(grad_x, "backward")
        return grad_x - 1


class MyModel(torch.nn.Module):
    def __init__(self, num_layer=4):
        super().__init__()
        self.num_layer = num_layer
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.linear(x)
        for i in range(self.num_layer):
            x = AFunction.apply(x)
        return x


model1 = MyModel(3)
model2 = MyModel(2)


optimizer = optim.AdamW(model1.parameters(), lr=1e-3)

label1 = torch.randint(0, 10, [1])
label2 = torch.randint(0, 10, [1])
loss_func = torch.nn.CrossEntropyLoss()

x = torch.randn([2, 10])
# with torch.no_grad():
if 1:
    tmp = model1(x)
# tmp.requires_grad = False
# tmp.retain_grad()
tmp1 = tmp[:1].detach()
tmp1.requires_grad = True

tmp2 = tmp[1:].detach()
tmp2.requires_grad = True
y1 = model2(tmp1)
y2 = model2(tmp2)
loss = loss_func(y1, label1)
loss.backward()
print(loss)
loss = loss_func(y2, label2)
loss.backward()
print("grads", tmp1.grad, tmp2.grad)
# print(tmp.grad)
tmp.backward(torch.cat([tmp1.grad, tmp2.grad]))
