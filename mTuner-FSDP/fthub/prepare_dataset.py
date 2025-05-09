import torch
import logging
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import transformers
from torch.utils.data import Dataset
from typing import Dict


def tokenize(prompt, cutoff_len, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, list_data_dict, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        # list_data_dict = list_data_dict[:data_length]
        # logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = (
            PROMPT_DICT["prompt_input"],
            PROMPT_DICT["prompt_no_input"],
        )
        # print(list_data_dict[0])
        if "instruction" in list_data_dict[0]:
            pass
        else:

            def get_input(query):
                if query.find("\n") == -1:
                    return ""
                return "\n".join(query.split("\n")[1:])

            list_data_dict = [
                {
                    "instruction": data["query"].split("\n")[0],
                    "input": get_input(data["query"]),
                    "output": data["response"],
                }
                for data in list_data_dict
            ]
        # import ipdb; ipdb.set_trace()
        sources = [
            (
                prompt_input.format_map(example)
                if example.get("input", "") != ""
                else prompt_no_input.format_map(example)
            )
            for example in list_data_dict
        ]
        targets = [
            f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict
        ]

        self.sources = sources
        self.targets = targets

    def __len__(self):
        return len(self.sources)

    def naive__getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __getitem__(self, i):
        # print(tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.unk_token_id)

        return dict(input_ids=self.sources[i], labels=self.targets[i])


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer, max_words=512):
        self.ann = data
        self.max_words = max_words
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]

        def get_input(query):
            if query.find("\n") == -1:
                return ""
            return "\n".join(query.split("\n")[1:])

        format = {
            "instruction": ann["query"].split("\n")[0],
            "input": get_input(ann["query"]),
            "output": ann["response"],
        }
        if ann.get("query", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(format)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(format)
        example = prompt + ann["response"]
        prompt = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example, dtype=torch.int64)
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        import copy

        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = -100
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
        }


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    text = f"Question: {example['query']}\n\nAnswer: {example['response']}"
    return text


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def prepare_dataset(rank, world_size, tokenizer, args):
    import datasets

    if args is not None:
        batch_size = args.train.batch_size
        seq_len = args.train.seq_len
    else:
        batch_size = 2
        seq_len = 128

    if (
        args is not None
        and hasattr(args, "filename")
        and args.filename.endswith("json")
    ):
        import json

        with open(args.filename) as f:
            list_data_dict = json.load(f)
        logging.warning("Loading data...")
        length = len(list_data_dict)
        dataset = {}
        splitter = int(length * 0.8)
        print("length", length)
        dataset["train"] = list_data_dict[:splitter]
        dataset["test"] = list_data_dict[splitter:]
        print(list_data_dict[0])
        exit()
    else:
        print(args)

    train_data = dataset["train"]
    valid_data = dataset["test"]
    logging.info(f"train dataset size {len(train_data)}")
    logging.info(f"valid dataset size {len(valid_data)}")

    train_dataset = InstructionDataset(
        train_data, tokenizer, max_words=args.train.seq_len
    )
    valid_dataset = InstructionDataset(
        valid_data, tokenizer, max_words=args.train.seq_len
    )

    train_sampler = DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=True
    )
    valid_sampler = DistributedSampler(
        valid_dataset, rank=rank, num_replicas=world_size
    )
    train_kwargs = {"batch_size": batch_size}
    test_kwargs = {"batch_size": batch_size}
    cuda_kwargs = {"num_workers": 2, "pin_memory": True, "shuffle": False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    # data_collator=transformers.DataCollatorForSeq2Seq(
    #         tokenizer, return_tensors="pt", padding=True)
    """datacollator会让训练变慢?"""
    train_loader = torch.utils.data.DataLoader(
        train_dataset, **train_kwargs, sampler=train_sampler, drop_last=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, **test_kwargs, sampler=valid_sampler, drop_last=True
    )
    """如果设置为True,这个是对最后的未完成的batch来说的,比如你的batch_size设置为64,而一个epoch只有100个样本,那么训练的时候后面的36个就被扔掉了.
    如果为False,那么会继续正常执行,只是最后的batch_size会小一点。"""
    return train_loader, valid_loader, train_sampler, valid_sampler


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from types import SimpleNamespace
    import torch.nn.functional as F

    logging.basicConfig(level=logging.INFO)
    model_name_or_path = "/mnt/data/zhongrx/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.float16
    ).to("cuda")
    try:
        model.to_bettertransformer()
    except:
        pass
    args = SimpleNamespace()
    args.train = SimpleNamespace()
    args.train.batch_size = 2
    args.train.seq_len = 128
    args.filename = "MetaMathQA-40K.json"
    train_loader, valid_loader, train_sampler, valid_sampler = prepare_dataset(
        0, 1, tokenizer, args
    )
    cnt = 0
    for batch in train_loader:
        print(
            batch["input_ids"].shape,
            batch["labels"].shape,
        )
        print(torch.max(batch["input_ids"]), torch.min(batch["input_ids"]))
        print(torch.max(batch["labels"]), torch.min(batch["labels"]))

        batch["labels"][batch["labels"] == -100] = 1
        print("input", tokenizer.decode(batch["input_ids"][0]))
        print("labels", tokenizer.decode(batch["labels"][0]))
        for key in batch.keys():
            batch[key] = batch[key].to("cuda")
        output = model(**batch, use_cache=False)
        print("loss", output["loss"])
        preds = F.softmax(output["logits"], dim=-1).argmax(dim=-1)
        print("output", tokenizer.decode(preds[0]))
        # for key in output.keys():
        #     print(key, output[key].shape)
        cnt += 1
        if cnt > 2:
            break
    print(len(train_loader))
