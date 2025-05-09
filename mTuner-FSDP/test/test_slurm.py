import logging
import torch
import fthub
import hydra
from omegaconf import DictConfig
import os
import torch.distributed as dist
import subprocess


def setup_distributed(backend="nccl", port=None):
    """
    Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" in os.environ:
            pass  # use MASTER_PORT in the environment variable
        else:
            os.environ["MASTER_PORT"] = "29500"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )

def cleanup():
    dist.destroy_process_group()

def run(rank, log_queue):
    setup_distributed()
    fthub.setup_worker_logging(rank, log_queue, is_slurm=True)
    logging.info(rank)
    cleanup()

@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(config: DictConfig):
    filename = (
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + "/main.log"
    )
    # torch.multiprocessing.set_start_method("spawn", force=True)
    log_queue = fthub.setup_primary_logging(filename, is_slurm=True)
    rank = int(os.environ["SLURM_PROCID"])
    run(rank, log_queue)
    return

if __name__ == "__main__":
    main()