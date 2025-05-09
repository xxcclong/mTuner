import torch


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(8192, 8192) for i in range(10)]
        )
        for it, item in enumerate(self.linears):
            if it % 2 == 0:
                item.weight.requires_grad = False
            print(item.weight.requires_grad)

    def forward(self, x):
        print(torch.cuda.memory_allocated())
        with torch.autograd.graph.save_on_cpu():
            for it, linear in enumerate(self.linears):
                x = linear(x)
                print(torch.cuda.memory_allocated())
        # for it, linear in enumerate(self.linears):
        #     if it % 2 == 0:
        #         x = linear(x)
        #     else:
        #         print("begin offload")
        #         with torch.autograd.graph.save_on_cpu():
        #             x = linear(x)
        #         print("end offload")
        #     print(torch.cuda.memory_allocated())
        return x
        # return self.linears(x)


m = MyModel().to("cuda")
optimizer = torch.optim.Adam(m.parameters(), lr=0.01)
a = torch.randn(10, 8192, device="cuda")
with torch.autograd.graph.save_on_cpu():
    b = m(a)
print(b)
loss = b.sum()
loss.backward()
optimizer.step()
# print(m.linears[0].weight.grad)
