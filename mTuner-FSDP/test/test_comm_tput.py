# from scalene import scalene_profiler

# scalene_profiler.stop()

import fthub
import torch
# import grpc
from fthub import FTHubServer, add_FTHubServerServicer_to_server, FTHubRequest2, FTHubReply2

import time

server_addr = f"octave:7877"
stub = fthub.FTHubServerStub(
    grpc.insecure_channel(server_addr,
                          options=[('grpc.max_send_message_length',
                                    fthub.MAX_MESSAGE_LENGTH),
                                   ('grpc.max_receive_message_length',
                                    fthub.MAX_MESSAGE_LENGTH)]))
message_tensor = torch.randn(400, 128, 768)
message = fthub.tensors2message((message_tensor, ))[0]
message_size_in_bytes = message_tensor.numel() * message_tensor.element_size()

response = stub.echo(FTHubRequest2(pid=0, tensors=message, forward=False))
# print(repr(response))
# scalene_profiler.start()
for i in range(10):
    t0 = time.time()
    response = stub.echo(FTHubRequest2(pid=0, tensors=message, forward=False))
    t1 = time.time()
    print("bandwidth",
          2 * message_size_in_bytes / (t1 - t0) / 1024 / 1024 / 1024, "GB/s")
# scalene_profiler.stop()