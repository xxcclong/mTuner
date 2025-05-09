import json
import logging

import os
import platform
import sys
import warnings
import array
import time

import torch
import numpy
import os

import pydot

from torch.profiler import profile, record_function, ProfilerActivity


class RunTimeMem:
    mem_grad = -1
    mem_activation = -1
    mem_base = -1
    mem_optimizer = -1


def draw_graph_normal_nodes(nodes, name):
    # colors = ["red", "blue", "yellow", "black"]
    dot_graph = pydot.Dot("my_graph", graph_type="graph")
    for node in nodes:
        dot_graph.add_node(
            pydot.Node(
                str(node.name),
            )
        )
    for node in nodes:
        for arg in node.args:
            if isinstance(arg, torch.fx.node.Node):
                dot_graph.add_edge(pydot.Edge(arg.name, node.name))
    dot_graph.write_pdf(f"{name}.pdf")


def draw_graph_grads_nodes(nodes, name):
    colors = ["red", "blue", "grey"]
    dot_graph = pydot.Dot("my_graph", graph_type="graph")
    for node in nodes:
        new_node = pydot.Node(str(node.name))
        if "tensor_meta" in node.meta and hasattr(
            node.meta["tensor_meta"], "requires_grad"
        ):
            # c = colors[int(node.meta["tensor_meta"].requires_grad)]
            new_node.set_fillcolor(colors[int(node.meta["tensor_meta"].requires_grad)])
            assert False
        else:
            new_node.set_fillcolor("blue")

        dot_graph.add_node(new_node)
    for node in nodes:
        for arg in node.args:
            if isinstance(arg, torch.fx.node.Node):
                dot_graph.add_edge(pydot.Edge(arg.name, node.name))
    dot_graph.write_pdf(f"{name}.pdf")


def draw_graph(nodes, name):
    colors = ["red", "blue", "yellow", "black"]
    dot_graph = pydot.Dot("my_graph", graph_type="graph")
    for node in nodes:
        dot_graph.add_node(
            pydot.Node(
                str(node.name),
                color=colors[int(node.status)],
                label="vid="
                + str(node.visit_id)
                + " s="
                + str(node.similarity)
                + " "
                + str(node.name)
                + "<->"
                + (str(node.pair_node.name) if node.pair_node else "None"),
            )
        )
    for node in nodes:
        for arg in node.args:
            if isinstance(arg, torch.fx.node.Node):
                dot_graph.add_edge(pydot.Edge(arg.name, node.name))
    dot_graph.write_pdf(f"{name}.pdf")


def do_bench(fn, num_trial=10):
    t0 = time.time()
    for _ in range(num_trial):
        fn()
        torch.cuda.synchronize()
    print(f"Time {str(fn)}: {(time.time() - t0) / num_trial}")


def do_bench_torch(fn, num_trial=10):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(num_trial):
            fn()
    # prof.export_chrome_trace("trace.json")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


send_size = 0
recv_size = 0


def message2tensors(props, names, tensors, device="cuda") -> dict:
    global recv_size
    ret = {}
    for it, name in enumerate(names):
        assert name in props, f"{name} {props.keys()}"
        prop = props[name]
        if prop.ttype == "tensor":
            t = torch.frombuffer(array.array("b", tensors[it]), dtype=prop.dtype).view(
                prop.shape
            )
            t = t.to(device)
            recv_size += t.numel() * t.element_size()
        elif prop.ttype == "shape":
            t = torch.frombuffer(array.array("b", tensors[it]), dtype=prop.dtype).view(
                prop.shape
            )
            t = torch.Size(t)
            recv_size += len(t)
        elif prop.ttype == "scalar":
            t = (
                torch.frombuffer(array.array("b", tensors[it]), dtype=prop.dtype)
                .view(prop.shape)
                .item()
            )
            recv_size += 1
        elif prop.ttype == "none":
            t = None
        else:
            assert False, f"unknown type {prop['type']}"
        ret[name] = t
    return ret


def tensors2message(tensors) -> list:
    global send_size
    ret = []
    if isinstance(tensors, tuple):
        indexes = range(len(tensors))
    elif isinstance(tensors, dict):
        indexes = tensors.keys()
    else:
        assert False, f"unknown type {type(tensors)}"
    # for name, t in tensors.items():  # FIXME: the order might be different
    for ind in indexes:
        t = tensors[ind]
        if isinstance(t, torch.Tensor):
            ret.append(t.detach().cpu().numpy().tobytes())
            send_size += t.numel() * t.element_size()
        elif isinstance(t, torch.Size):
            ret.append(numpy.array(list(t)).tobytes())
            send_size += len(t)
        elif isinstance(t, int):
            ret.append(numpy.array([t]).tobytes())
            send_size += 1
        elif t is None:
            ret.append(numpy.array([0]).tobytes())
        else:
            assert False, f"unknown type {type(t)} {ind}"
    return ret


def log_size():
    global send_size, recv_size
    print(f"send_size={send_size}, recv_size={recv_size}")
    send_size = 0
    recv_size = 0


def get_size():
    return send_size, recv_size


class TensorProps(object):
    def __init__(self, ttype, shape, dtype=None, device=None):
        self.ttype = ttype
        self.shape = shape
        self.dtype = dtype
        self.device = device

    def __str__(self):
        return f"{self.ttype} {self.shape} {self.dtype} {self.device}"


def tensor2prop(t):
    if isinstance(t, torch.Size):
        return TensorProps(ttype="shape", shape=len(t), dtype=torch.int64, device="cpu")
    elif isinstance(t, torch.Tensor):
        return TensorProps(ttype="tensor", shape=t.shape, dtype=t.dtype, device="cuda")
    elif isinstance(t, int):
        return TensorProps(ttype="scalar", shape=1, dtype=torch.int64, device="cpu")
    elif isinstance(t, float):
        return TensorProps(ttype="scalar", shape=1, dtype=torch.float32, device="cpu")
    elif t is None:
        return TensorProps(ttype="none", shape=0, dtype=torch.int32, device="cpu")
    else:
        raise NotImplementedError(f"Unknown type {type(t)}")


def find_max_batch_size(handles):
    memories = [item[0] for item in handles[0].memory_traces]
    max_usage = max(memories)
    # min_usage = memories[-1]
    min_usage = RunTimeMem.mem_base
    overhead = 8e9
    total_memory = 4e10
    grad = RunTimeMem.mem_grad
    assert grad != -1, "gradient memory not profiled"
    logging.info(
        f"activation per sample: {max_usage - min_usage} base usage {min_usage} gradient usage {grad}"
    )
    return int((total_memory - overhead - min_usage - grad) // (max_usage - min_usage))


def log_memory_trace(
    handles,
    base_time,
    config,
):
    model_name = config.model.strip("/").split("/")[-1]
    batch_size = config.train.batch_size
    seq_len = config.train.seq_len
    is_peft = config.is_peft
    skip_comm = config.debug.skip_comm
    ac = config.ac
    time_dim_layer_id = config.time_dim.layer_id
    peft_str = "peft" if is_peft else "full"
    skip_str = "commed" if not skip_comm else "skip"
    ac_str = ac
    time_dim_str = f"timedim{time_dim_layer_id}"
    min_param_str = f"min_param{config.min_param}"
    filename = f"memory_traces/{model_name}_{config.world_size}_{batch_size}_{seq_len}_{peft_str}_{skip_str}_{ac_str}_{time_dim_str}_{min_param_str}.txt"
    with open(filename, "w") as f:
        for item in handles[0].memory_traces:
            f.write(str(item[0]) + " " + str(item[1] - base_time) + "\n")
    logging.info(filename)
    return
    with open(f"inter_train_size_{model_name}.txt", "w") as f:
        for handle in handles:
            # print(handle.memory_state)
            f.write(
                str(handle.handle_init_id)
                + "\t"
                + str(handle.memory_state[f"start_unshard_HandleTrainingState.FORWARD"])
                + "\n"
            )
            f.write(
                str(handle.handle_init_id)
                + "\t"
                + str(handle.memory_state[f"end_unshard_HandleTrainingState.FORWARD"])
                + "\n"
            )
            f.write(
                str(handle.handle_init_id)
                + "\t"
                + str(handle.memory_state[f"start_reshard_HandleTrainingState.FORWARD"])
                + "\n"
            )
            f.write(
                str(handle.handle_init_id)
                + "\t"
                + str(handle.memory_state[f"end_reshard_HandleTrainingState.FORWARD"])
                + "\n"
            )
        for handle in reversed(handles):
            if "start_unshard_HandleTrainingState.BACKWARD_PRE" in handle.memory_state:
                f.write(
                    str(handle.handle_init_id)
                    + "\t"
                    + str(
                        handle.memory_state[
                            "start_unshard_HandleTrainingState.BACKWARD_PRE"
                        ]
                    )
                    + "\n"
                )
                f.write(
                    str(handle.handle_init_id)
                    + "\t"
                    + str(
                        handle.memory_state[
                            "end_unshard_HandleTrainingState.BACKWARD_PRE"
                        ]
                    )
                    + "\n"
                )
            if "start_reshard_HandleTrainingState.BACKWARD_PRE" in handle.memory_state:
                f.write(
                    str(handle.handle_init_id)
                    + "\t"
                    + str(
                        handle.memory_state[
                            "start_reshard_HandleTrainingState.BACKWARD_PRE"
                        ]
                    )
                    + "\n"
                )
                f.write(
                    str(handle.handle_init_id)
                    + "\t"
                    + str(
                        handle.memory_state[
                            "end_reshard_HandleTrainingState.BACKWARD_PRE"
                        ]
                    )
                    + "\n"
                )
