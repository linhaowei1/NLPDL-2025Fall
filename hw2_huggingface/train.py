import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
# import torch_npu ## Uncomment this line if you are using Ascend NPU
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    ## TODO: import other classes you need
)
import evaluate
import peft
import adapters
import wandb
from dataHelper import get_dataset

# os.environ["WANDB_MODE"] = "offline" ## Uncomment this line if you cannot connect to wandb server


@dataclass
class BaseArgs:
    ## TODO: define your arguments here and set default values or fields
    dataset: None
    model_name: None
    sep_token: None  # default "<SEP>"
    peft: None
    max_length: None  # default 256


@dataclass
class LoraArgs:
    ## TODO: define your arguments here and set default values or fields
    rank: int
    alpha: int
    dropout: float


logger = logging.getLogger(__name__)


def print_trainable_parameters(model):  # for generic use
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params:,} || "
          f"all params: {all_params:,} || "
          f"trainable%: {100 * trainable_params / all_params:.2f}%")


def parse_arguments():
    ## TODO: parse arguments
    parser = None  # use HfArgumentParser
    raise NotImplementedError("Argument parsing not implemented yet!")


def set_random_seed(seed: int):
    ## TODO: set random seed
    raise NotImplementedError("Random seed setting not implemented yet!")


def set_logger():
    ## TODO: set up logging such that the log messages are printed to stdout
    raise NotImplementedError("Logger setup not implemented yet!")


def load_data(dataset_name: str, sep_token: str = "<SEP>"):
    ## TODO: load dataset using `get_dataset()`
    raw_dataset = None  # use get_dataset
    num_labels = None  # set the number of labels according to dataset
    return raw_dataset, num_labels


def get_model(model_name: str, num_labels: int):
    ## TODO: get model and tokenizer
    config = None  # use AutoConfig
    tokenizer = None  # use AutoTokenizer
    model = None  # use AutoModelFor???

    ## TODO: set pad_token for tokenizer and model (Why?)

    return tokenizer, model


def tokenize_data(raw_dataset, tokenizer, max_length: int = 256):
    ## TODO: tokenize dataset using tokenizer and dataset.map()
    train_dataset, eval_dataset = None, None
    return train_dataset, eval_dataset


def get_lora_model(model, lora_args):
    ## TODO: use LoRA to wrap your model
    model = None
    model.print_trainable_parameters()
    return model


def get_adapter_model(model, adapter_args):
    ## TODO: use Adapter to wrap your model
    model = None
    print_trainable_parameters(model)
    return model


def get_data_collator(tokenizer):
    ## TODO: define data collator
    data_collator = None
    return data_collator


def compute_metrics(p):
    ## TODO: compute accuracy, macro F1 and micro F1
    return {
        "accuracy": None,
        "macro_f1": None,
        "micro_f1": None,
        "weight_f1": None
    }


def get_trainer():
    ## TODO: define Trainer with appropriate arguments
    trainer = None
    return trainer


def main():
    base_args, train_args, adapter_args, lora_args = parse_arguments()

    set_random_seed(train_args.seed)

    set_logger()

    raw_dataset, num_labels = load_data(base_args.dataset, base_args.sep_token)

    tokenizer, model = get_model(base_args.model_name, num_labels)

    train_result, eval_result = tokenize_data(raw_dataset, tokenizer,
                                              base_args.max_length)

    # peft method
    if base_args.peft != None:
        if base_args.peft.lower() == "lora":
            model = get_lora_model(model, lora_args)
        elif base_args.peft.lower() == "adapter":
            model = get_adapter_model(model, adapter_args)
        else:
            raise ValueError("Unsupported PEFT method!")

    ## TODO: define data collator
    data_collator = get_data_collator(tokenizer)

    ## TODO: initialize wandb and set config
    wandb.init()  # add your project and record name
    wandb.config = {
        "epoch": None,
        "batch_size": None,
        "lr": None
    }  # set your config

    if train_args.do_train:
        logger.info("*** Train ***")
        ## TODO: define Trainer and start training
        trainer = get_trainer()  ## TODO: implement get_trainer() function
        train_result = None
        metrics = None

        ## TODO: save model, state and metricss

    if train_args.do_eval:
        logger.info("*** Evaluation ***")

        ## TODO: run evaluation and get metrics
        metrics = None

        ## TODO: log metrics

    if train_args.do_predict:
        logger.info("*** Predict ***")
        ## TODO: predict without checking, and save the results to `predict_results.txt`


if __name__ == "__main__":
    main()
