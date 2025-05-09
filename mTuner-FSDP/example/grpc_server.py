from fthub import FTHubServer, add_FTHubServerServicer_to_server, FTHubRequest, FTHubReply, FTHubRequest2, FTHubReply2
from fthub import message2tensors, tensors2message, TrainingMode

import array, torch
import asyncio, grpc
import logging
import time
import argparse

# logging.basicConfig(level=logging.INFO)

from fthub import get_model
from fthub import MAX_MESSAGE_LENGTH


class FTHubService(FTHubServer):

    def __init__(self, dev_id):
        t0 = time.time()
        self.dev_id = dev_id
        if self.dev_id >= 0:
            self.model = get_model(is_server=True).cuda(self.dev_id)
        else:
            self.model = get_model(is_server=True)
        self.registered_clients = set()
        print(f"server finish init in {(time.time() - t0):.2f} s")

    # async def init(self, request: InitRequest, context) -> InitReply:
    #     name = request.name
    #     client_id = request.client_id
    #     shared_ids = request.shared_ids
    #     if name.lower() == "gpt2":
    #         model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    #     elif "neo" in name.lower():
    #         from transformers import GPTNeoForCausalLM
    #         model = GPTNeoForCausalLM.from_pretrained(
    #             "EleutherAI/gpt-neo-1.3B").to(device)
    #     else:
    #         try:
    #             model = AutoModelForCausalLM.from_pretrained(name).to(device)
    #         except:
    #             raise NotImplementedError

    #     self.model = get_model(is_server=True)
    #     return InitReply(status=1)

    async def echo(self, request: FTHubRequest2, context) -> FTHubReply2:
        return FTHubReply2(tensors=request.tensors)

    async def ftserve(self, request: FTHubRequest, context) -> FTHubReply:
        pid = request.pid
        tensors = request.tensors
        forward = request.forward
        training = TrainingMode(request.training)
        t0 = time.time()
        if forward:
            inputs = message2tensors(self.model.input_props,
                                     self.model.partitions[pid].input_names,
                                     tensors)
            t1 = time.time()
            outputs = self.model.forward_single_partition(
                pid, inputs, training=TrainingMode)
            t2 = time.time()
        else:
            grads = message2tensors(self.model.grad_props,
                                    self.model.partitions[pid].output_names,
                                    tensors)
            t1 = time.time()
            outputs = self.model.backward_single_partition(pid, grads)
            t2 = time.time()
        # print("sending back", outputs.keys(), [
        #     x.shape if isinstance(x, torch.Tensor) else f"non-tensor_{x}"
        #     for x in outputs.values()
        # ])
        message = tensors2message(outputs)
        t3 = time.time()

        fb = "fwd" if forward else "bwd"

        print(f"{fb}\tunpack\t{pid}\t{(t1 - t0):.4f}")
        print(f"{fb}\tcomp\t{pid}\t{(t2 - t1):.4f}")
        print(f"{fb}\tpack\t{pid}\t{(t3 - t2):.4f}")

        return FTHubReply(tensors=message)


async def serve(args):
    server = grpc.aio.server(
        options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH
                  ), ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])
    add_FTHubServerServicer_to_server(FTHubService(args.gpu), server)
    # using ip v6
    addr = "0.0.0.0:7877"
    server.add_insecure_port(addr)
    logging.info(f"Starting server on {addr}")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(serve(args))
