from fthub import get_model
import torch
from fthub.util import do_bench


def test_generate():
    from transformers import GPTNeoForCausalLM, GPT2Tokenizer
    model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").cuda()
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    prompt = "Hello, my dog is cute"
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(inputs.cuda())
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))


def test_backward():
    p1, p2 = get_model()
    dummy_input = torch.randint(0, 100, (4, 512), device="cuda")
    loss_fn = torch.nn.MSELoss()
    o1 = p1.forward({"input_ids": dummy_input})
    o1 = o1["output"]
    label = torch.randn_like(o1)

    for _ in range(10):
        o1 = p1.forward({"input_ids": dummy_input})["output"]
        loss = loss_fn(o1, label)
        print(loss.item())
        loss.backward()


def test_prof_exec():
    p1, p2 = get_model()
    dummy_input = torch.randint(0, 100, (4, 512), device="cuda")
    do_bench(lambda: p1.forward({"input_ids": dummy_input}))
    do_bench(lambda: p2.forward({"input_ids": dummy_input}))


def test_single_backward():
    p1, p2 = get_model()
    dummy_input = torch.randint(0, 100, (4, 512), device="cuda")
    # draw_graph_normal_nodes(p1.partitions[0].gm.graph.nodes, "error")
    pid = 0
    optimizer = torch.optim.Adam(p1.partitions[pid].gm.parameters(), lr=1e-3)
    outputs = p1.forward_single_partition(pid, {"input_ids": dummy_input})
    grads = {}
    for k in outputs:
        if isinstance(outputs[k], torch.Tensor):
            grads[k] = torch.ones_like(outputs[k])
            print(k, grads[k].shape)
        else:
            grads[k] = None

    for _ in range(10):
        outputs = p1.forward_single_partition(pid, {"input_ids": dummy_input})
        p1.backward_single_partition(pid, grads)
        print(outputs["transformer_h_0_ln_1"].sum())
        # outputs["transformer_h_0_ln_1"].backward(grad)
        optimizer.step()


if __name__ == "__main__":
    test_prof_exec()
    test_generate()

    # test gradient
    test_backward()
    test_single_backward()
