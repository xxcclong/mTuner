import re

class ModelConfig:
    def __init__(self, seq_len, hidden_size, num_heads, num_query_groups, num_layers, ffn_hidden_size, norm_eps, model_size):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_query_groups = num_query_groups
        self.num_layers = num_layers
        self.ffn_hidden_size = ffn_hidden_size
        self.norm_eps = norm_eps
        self.weight_size = 2 * (self.hidden_size * self.ffn_hidden_size * 3 + self.hidden_size * (self.hidden_size // self.num_heads) * (self.num_heads + self.num_query_groups * 2))
        self.model_size = model_size
        self.act_size = self.hidden_size # TODO
        self.rec_act_size = self.hidden_size # TODO
        # print(f"{self.model_size=} {self.weight_size / 1e9=}")

def parse_slurm_log_last_occurrence(log_file_path):
    """
    Parse a Slurm log file and extract the LAST occurrence of:
    - Execution time per iteration
    - Memory usage (before-forward and after-loss)
    
    Args:
        log_file_path (str): Path to the Slurm log file
        
    Returns:
        dict: Parsed data containing:
            - 'iteration_time_ms' (float)
            - 'memory_before_forward_mb' (float)
            - 'memory_after_loss_mb' (float)
    """
    data = {
        'iteration_time_ms': None,
        'memory_before_forward_mb': None,
        'memory_after_loss_mb': None
    }
    
    # Regular expressions for pattern matching
    time_pattern = r'elapsed time per iteration \(ms\): (\d+\.\d+)'
    mem_before_pattern = r'rank \d+ before-forward Memory used (\d+\.\d+) MB'
    mem_after_pattern = r'rank \d+ after-loss Memory used (\d+\.\d+) MB'
    
    with open(log_file_path, 'r') as f:
        for line in f:
            # Check for iteration time (update if found)
            time_match = re.search(time_pattern, line)
            if time_match:
                data['iteration_time_ms'] = float(time_match.group(1))
            
            # Check for memory before forward (update if found)
            mem_before_match = re.search(mem_before_pattern, line)
            if mem_before_match:
                data['memory_before_forward_mb'] = float(mem_before_match.group(1))
            
            # Check for memory after loss (update if found)
            mem_after_match = re.search(mem_after_pattern, line)
            if mem_after_match:
                data['memory_after_loss_mb'] = float(mem_after_match.group(1))
    # print(data, log_file_path) 
    return data['iteration_time_ms'], data['memory_before_forward_mb'], data['memory_after_loss_mb']

class Implementation:

    # batch_size, tp size, fsdp size, recompute, is shared
    # base partition, peak partition, valley partition

    def __init__(self, model_config, gather_to, discard_to, bwd_discard_to, recompute, batch_size):
        self.valid = True
        self.model_config = model_config
        self.gather_to = gather_to
        self.discard_to = discard_to
        self.bwd_discard_to = bwd_discard_to
        self.tp_size = 8 // self.gather_to
        self.dp_size = 8 // self.tp_size
        # self.fsdp_size = fsdp_size
        self.recompute = recompute
        self.batch_size = batch_size
        self.path = f"logs/gpt3_train_{self.model_config.seq_len}_{self.tp_size}_1_{self.model_config.model_size}.log"
        t, mem_before, mem_after = parse_slurm_log_last_occurrence(self.path)
        if t is None:
            t = 100000
            mem_after = 10000
            mem_before = 0
            self.valid = False
        layer = 20
        act_mem = (mem_after - mem_before) / layer / 1e3 * batch_size
        if recompute:
            act_mem /= 8
            t *= 1.5
        if self.gather_to != self.bwd_discard_to:
            t *= 1.05
        if self.discard_to != self.bwd_discard_to:
            t *= 1.05
        self.peak_memory = self.cal_peak() + act_mem
        self.valley_memory = self.cal_valley() 
        self.exec_time = t / batch_size / layer / self.dp_size
        # print(f"{self.peak_memory=} {self.cal_peak()=} {self.valley_memory=} {self.exec_time=}")

    def cal_peak(self):
        base_weight = self.model_config.weight_size
        w = base_weight * (self.discard_to / 8)
        return w * 2 / 1e9

    def cal_valley(self):
        base_weight = self.model_config.weight_size
        w = base_weight * (self.bwd_discard_to / 8)
        return w * 2 / 1e9

    def __str__(self):
        return f"{self.gather_to=} {self.discard_to=} {self.bwd_discard_to=} {self.recompute=} {self.batch_size=} {self.peak_memory=} {self.valley_memory=} {self.exec_time=}"

class Result:
    def __init__(self, batch_size, tp_size, impls, ):
        self.batch_size = batch_size
        self.tp_size = tp_size
        self.impls = impls
