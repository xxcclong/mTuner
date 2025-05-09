import logging
from tqdm import tqdm
import torch
import time
from torch.profiler import ProfilerActivity
from torch.profiler import profile, record_function
import time
import torch.distributed as dist
import torch.nn.functional as F
from .fsdp import FullyShardedDataParallel as FSDP
from .fsdp import FullStateDictConfig
from .fsdp import StateDictType
from .util import log_memory_trace, RunTimeMem


def get_peft_weights(model):
    save_policy = FullStateDictConfig(offload_to_cpu=False, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        states = model.state_dict()
    peft_states = {}
    for k, v in states.items():
        if "lora" in k:
            peft_states[k] = v
    return peft_states


def save_peft_weights(model, rank, save_name):
    t_save = time.time()
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = model.state_dict()
    if rank == 0:
        save_state = {}
        for k, v in cpu_state.items():
            if "lora" in k:
                save_state[k] = v
        torch.save(save_state, save_name)
    if rank == 0:
        logging.info(f"save time {time.time() - t_save}")


def run_train_with_data(
    model,
    optimizer,
    scheduler,
    rank,
    epoch,
    loader,
    sampler,
    batch_size,
    grad_accumulation_steps,
    log_dir,
    tokenizer,
    args,
):
    model.train()
    fsdp_loss = torch.zeros(2, device=rank)
    sampler.set_epoch(epoch)
    if rank == 0:
        inner_pbar = tqdm(
            range(len(loader) * batch_size),
            colour="blue",
            bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}",
        )
    for it, batch in enumerate(loader):
        for key in batch.keys():
            batch[key] = batch[key].to(rank)
        output = model(
            **batch,
            use_cache=False,  # reduce
        )
        # preds = F.softmax(output["logits"], dim=-1).argmax(dim=-1)
        # input_str = tokenizer.batch_decode(list(batch["input_ids"].cpu().numpy()))
        # output_str = tokenizer.batch_decode(list(preds.cpu().numpy()))
        # if rank == 0:
        #     logging.info(
        #         f"input: {input_str}\noutput: {output_str}\nlearning rate: {optimizer.param_groups[0]['lr']}"
        #     )
        loss = output["loss"]
        del output
        loss = loss / grad_accumulation_steps
        loss.backward()
        if (it + 1) % grad_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        fsdp_loss[0] += loss.item()
        if rank == 0:
            logging.info(f"loss: {loss} {batch_size}")
            loss_value = loss.item()
        fsdp_loss[1] += batch_size
        del loss
        if rank == 0:
            inner_pbar.update(batch_size)
        if (
            args.train.save_interval > 0
            and it % args.train.save_interval == 0
            and it != 0
        ):  # save model weights
            save_peft_weights(model, rank, f"{log_dir}/{epoch}_{it}_{loss_value}.pt")
        if fsdp_loss[1] == len(loader) * batch_size:
            break

    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    train_acc = fsdp_loss[0] / fsdp_loss[1]
    if rank == 0:
        logging.info(f"epoch {epoch} avg train loss {train_acc}")
        inner_pbar.close()
    return train_acc


def run_train(
    num_iter, batch_size, seq_len, model, loss_func, optimizer, rank, handles, args
):
    t_overall_0 = time.time()
    if rank == 0:
        logging.info(f"before training {torch.cuda.memory_allocated(rank)}")
    for i in range(num_iter):
        for handle in handles:
            handle.set_time_measurement()
        torch.cuda.synchronize(rank)
        dist.barrier()
        t0 = time.time()
        input_ids = torch.randint(0, 1024, (batch_size, seq_len)).to(rank)
        output = model(
            input_ids,
            labels=input_ids,
            use_cache=False,  # reduce
        )
        loss = loss_func(
            output.logits.view(-1, output.logits.size(-1)),
            input_ids.view(-1),
        )
        mem = torch.cuda.memory_allocated(rank)
        if rank == 0:
            logging.info(f"peak memory usage {mem}")
        del output
        torch.cuda.synchronize(rank)
        dist.barrier()
        tmid = time.time()
        loss.backward()
        if RunTimeMem.mem_grad == -1:
            mem_before_opt_step = torch.cuda.memory_allocated(rank)
        mem = max(mem, torch.cuda.memory_allocated(rank))
        optimizer.step()
        optimizer.zero_grad()
        if RunTimeMem.mem_grad == -1:
            RunTimeMem.mem_grad = (
                torch.cuda.memory_allocated(rank) - mem_before_opt_step
            )
        del loss
        torch.cuda.synchronize(rank)
        dist.barrier()
        if RunTimeMem.mem_base == -1:
            RunTimeMem.mem_base = torch.cuda.memory_allocated(rank)
        t1 = time.time()
        if rank == 0:
            logging.info(f"after-optimizer {torch.cuda.memory_allocated(rank)}")
        if rank == 0:
            logging.info(
                f"rank {rank} iter {i} fwd {tmid - t0} bwd {t1 - tmid} time {t1 - t0} mem {mem} tput {args.train.seq_len * batch_size * args.world_size / (t1 - t0)}"
            )
        num_unshard = 0
        num_allgather = 0
        allgather_size = 0
        comm_t = 0
        for handle in handles:
            num_unshard += handle.unshard_cnt
            num_allgather += handle.unshard_allgather
            allgather_size += handle.allgather_size
            handle.unshard_cnt = 0
            handle.unshard_allgather = 0
            handle.allgather_size = 0
            comm_t += handle.get_time_measurement()
        if rank == 0:
            logging.info(
                f"rank {rank} iter {i} num_unshard {num_unshard} num_allgather {num_allgather} allgather_size {allgather_size} allgather_time {comm_t}"
            )

    t_overall_1 = time.time()
    if rank == 0 and num_iter > 2:
        logging.info(
            f"overall_tput {args.train.seq_len * batch_size * args.world_size * num_iter / (t_overall_1 - t_overall_0)} avg_time {(t_overall_1 - t_overall_0) / num_iter} batch_size {batch_size} seq_len {seq_len} num_iter {num_iter}"
        )


def run_train_dual_model(
    num_iter,
    batch_size,
    sub_batch_size,
    seq_len,
    model1,
    model2,
    loss_func,
    optimizer,
    rank,
    handles,
    args,
):
    t_overall_0 = time.time()
    for i in range(num_iter):
        torch.cuda.synchronize(rank)
        dist.barrier()
        t0 = time.time()
        if rank == 0:
            logging.info(f"before training {torch.cuda.memory_allocated(rank)}")
        input_ids = torch.randint(0, 1024, (batch_size, seq_len)).to(rank)
        hidden_states = model1(input_ids)
        hidden_states_grad = []
        if rank == 0:
            logging.info(f"before repeated {torch.cuda.memory_allocated(rank)}")
        for j in range(0, batch_size, sub_batch_size):
            tmp = hidden_states[j : j + sub_batch_size].detach()
            tmp.requires_grad = True
            output = model2(tmp)
            loss = loss_func(
                output.view(-1, output.size(-1)),
                input_ids[j : j + sub_batch_size].view(-1),
            )
            mem = torch.cuda.memory_allocated(rank)
            if rank == 0:
                logging.info(f"peak memory usage {mem}")
            del output
            loss.backward()
            del loss
            hidden_states_grad.append(tmp.grad)
        hidden_states.backward(torch.cat(hidden_states_grad))
        del hidden_states
        del hidden_states_grad
        if RunTimeMem.mem_grad == -1:
            mem_before_opt_step = torch.cuda.memory_allocated(rank)
        optimizer.step()
        optimizer.zero_grad()
        if RunTimeMem.mem_grad == -1:
            RunTimeMem.mem_grad = (
                torch.cuda.memory_allocated(rank) - mem_before_opt_step
            )
        if rank == 0:
            logging.info(f"after-optimizer {torch.cuda.memory_allocated(rank)}")
        # if rank == 0:
        #     logging.info(loss.item())  # print loss for validation
        torch.cuda.synchronize(rank)
        dist.barrier()
        t1 = time.time()
        if rank == 0:
            logging.info(f"rank {rank} iter {i} time {t1 - t0}")
        for handle in handles:
            handle.unshard_cnt = 0
    t_overall_1 = time.time()
    if rank == 0 and num_iter > 2:
        logging.info(
            f"overall_tput {args.train.seq_len * batch_size * args.world_size * num_iter / (t_overall_1 - t_overall_0)} avg_time {(t_overall_1 - t_overall_0) / num_iter} batch_size {batch_size} seq_len {seq_len} num_iter {num_iter}"
        )


def run_train_pipe(
    num_iter, batch_size, seq_len, pipe_model, loss_func, optimizer, world_size, args
):
    t_overall_0 = time.time()
    for i in range(num_iter):
        t0 = time.time()
        input_ids = torch.randint(0, 1024, (batch_size, seq_len), device=0)
        output = pipe_model(input_ids).local_value()
        loss = loss_func(
            output.view(-1, output.size(-1)),
            input_ids.view(-1).to(output.device),
        )
        del output
        logging.info("memory usage {}".format(torch.cuda.memory_allocated(0)))
        loss.backward()
        if RunTimeMem.mem_grad == -1:
            mem_before_opt_step = torch.cuda.memory_allocated(0)
        optimizer.step()
        optimizer.zero_grad()
        if RunTimeMem.mem_grad == -1:
            RunTimeMem.mem_grad = torch.cuda.memory_allocated(0) - mem_before_opt_step
        del loss
        t1 = time.time()
        logging.info(f"iter {i} time {t1 - t0}")
    t_overall_1 = time.time()
    logging.info(
        f"overall_tput {args.train.seq_len * batch_size * num_iter / (t_overall_1 - t_overall_0)} avg_time {(t_overall_1 - t_overall_0) / num_iter} batch_size {batch_size} seq_len {seq_len} num_iter {num_iter}"
    )


def profile(model, handles, loss_func, optimizer, scheduler, args, do_profile=False):
    num_iter = 5
    run_dual_model = args.dual_model.num_layer > 0 and args.dual_model.batch_size > 0
    base_time = time.time()

    if do_profile:
        for handle in handles:
            handle.set_profile(batch_size=args.train.batch_size)
    if args.pipeline.use_pipeline:
        run_train_pipe(
            num_iter,
            args.train.batch_size,
            args.train.seq_len,
            model,
            loss_func,
            optimizer,
            args.world_size,
            args,
        )
    elif run_dual_model:
        run_train_dual_model(
            num_iter,
            args.train.batch_size,
            args.dual_model.batch_size,
            args.train.seq_len,
            model[0],
            model[1],
            loss_func,
            optimizer,
            args.rank,
            handles,
            args,
        )
    else:
        run_train(
            num_iter,
            args.train.batch_size,
            args.train.seq_len,
            model,
            loss_func,
            optimizer,
            args.rank,
            handles,
            args,
        )
    if do_profile:
        for handle in handles:
            handle.finish_profile()
        if len(handles) > 0 and hasattr(handles[0], "handle_visit_id"):
            handles = sorted(handles, key=lambda handle: handle.handle_visit_id)
        if args.rank == 0:
            log_memory_trace(handles, base_time, args)


def train(
    model,
    tokenizer,
    train_loader,
    train_sampler,
    loss_func,
    optimizer,
    scheduler,
    log_dir,
    args,
):
    if tokenizer is not None and args.with_data:
        if args.rank == 0:
            logging.info("start training")
        for epoch in range(args.train.epoch):
            train_acc = run_train_with_data(
                model,
                optimizer,
                scheduler,
                args.rank,
                epoch,
                train_loader,
                train_sampler,
                args.train.batch_size,
                args.train.grad_accumulation_steps,
                log_dir.replace("main.log", ""),  # directory to store checkpoint
                tokenizer,
                args,
            )
            if args.rank == 0:
                logging.info(f"epoch {epoch} train acc {train_acc}")
