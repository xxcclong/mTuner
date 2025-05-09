import torch
from torch.fx import symbolic_trace
from fthub import comm_op
from typing import Dict

if __name__ == "__main__":
    # define a torch model
    class MyModel(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)
            self.linear2 = torch.nn.Linear(10, 10)

        def forward(self, x, y):
            z = x
            c = self.linear2(z[1] + self.linear((x + y) * 2))
            d = c + 1
            return d

    model = MyModel()
    gm = symbolic_trace(model)
    print(gm)
    # for node in gm.graph.nodes:
    #     print(node, node.op, node.target, node.args, node.kwargs)

    # change the name of a node
    for node in gm.graph.nodes:
        if node.name == "x":
            node.name = "xx"
        break
        # if node.name == "c":
        #     break
    gm.recompile()
    print(gm)

    # change the output of graph
    new_graph = torch.fx.Graph()
    env: Dict[torch.fx.Node, torch.fx.Node] = {}
    for node in gm.graph.nodes:
        if node.op == "output":
            continue
        new_node = new_graph.node_copy(node, lambda x: env[x])
        if node.name == "add_1":
            return_node = new_node
        env[node] = new_node
    new_graph.output(return_node)
    new_graph.lint()
    # fx_model = gm.graph.fx_graph.owning_module
    new_gm = torch.fx.GraphModule(gm, new_graph)
    print(new_gm)

    # gm.graph.output = node
    # gm.output = node
    # gm.recompile()
    # print(node)
    # print(gm)