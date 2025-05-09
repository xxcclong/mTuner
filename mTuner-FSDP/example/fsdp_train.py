import os
import hydra
from omegaconf import DictConfig
import logging

import torch
import torch.multiprocessing as mp


import fthub


is_slurm = "SLURM_JOB_ID" in os.environ and int(os.environ.get("SLURM_NTASKS", -1)) > 1


def fsdp_main(rank, world_size, args, log_queue, log_dir):
    args.rank = rank
    args.world_size = world_size
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    fthub.setup_worker_logging(rank, log_queue, is_slurm=True)
    if rank == 0:
        logging.info(args)
    fthub.setup_distributed(is_slurm=is_slurm)
    fthub.set_validation(args.validation)
    model, tokenizer = fthub.get_model_and_tokenizer(args)

    model, optimizer, scheduler, handles = fthub.init_model(model, args)
    # use valley optimization for better memory utilization
    fthub.init_partial_process_group(args, handles)

    loss_func = torch.nn.CrossEntropyLoss()

    # Find out maximum batch size by profiling first
    need_max_batch_size = args.train.batch_size == "max"
    if need_max_batch_size:
        args.train.batch_size = 1

    # Run profiling
    if need_max_batch_size:
        fthub.profile(
            model, handles, loss_func, optimizer, scheduler, args, do_profile=True
        )

    if need_max_batch_size:
        args.train.batch_size = fthub.util.find_max_batch_size(handles)
        if rank == 0:
            logging.info(f"max batch size {args.train.batch_size}")

    # dataset initialization
    if tokenizer is not None and args.with_data:
        train_loader, val_loader, train_sampler, val_sampler = fthub.prepare_dataset(
            rank, world_size, tokenizer, args
        )
        # tokenizer.add_special_tokens({'pad_token': '<PAD>'})
        tokenizer.pad_token_id = 0
        fthub.train(
            model,
            tokenizer,
            train_loader,
            train_sampler,
            loss_func,
            optimizer,
            scheduler,
            log_dir,
            args,
        )
    else:
        fthub.profile(
            model, handles, loss_func, optimizer, scheduler, args, do_profile=False
        )

    fthub.cleanup()


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(config: DictConfig):
    filename = (
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + "/main.log"
    )
    log_queue = fthub.setup_primary_logging(filename, is_slurm=is_slurm)
    if not is_slurm:
        torch.multiprocessing.set_start_method("spawn", force=True)
        world_size = torch.cuda.device_count()
        mp.spawn(
            fsdp_main, args=(world_size, config, log_queue, filename), nprocs=world_size
        )
    else:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        fsdp_main(rank, world_size, config, log_queue, filename)

    logging.info(filename)


if __name__ == "__main__":
    main()
