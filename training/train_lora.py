import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
import transformers
from transformers import Trainer, DataCollatorForLanguageModeling, TrainingArguments, HfArgumentParser
from datasets import load_dataset
from datetime import datetime as dt


@dataclass
class LoraArguments:
    lora_r: int = field(
        default=2,
        metadata={
            "help": "Lora attention dimensions. Corresponds to the number of parameters. " 
            "The paper demostrates a low rank and for a low rank and adapt more weight adapt more weight matrices."
        }
    )
    lora_alpha: int = field(
        default=4,
        metadata={"help": "he alpha parameter for Lora scaling. usually double rank."}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout for probability for the Lora layer."}
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: ['q_proj','k_proj','v_proj','o_proj'],
        metadata={"help": "which weight matrices to adapt. The paper argues more matricies with lower ranks."}
    )
    bias: str = field(
        default="none",
        metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"}
    )
    

@dataclass
class ModelArguments:
    base_model: Optional[str] = field(default="huggyllama/llama-7b")


@dataclass
class DataArguments:
    data_path: str = field(
        default='../data/top.jsonl', 
        metadata={"help": "Path to the training data."}
    )
    max_samples: int = field(
        default=None,
        metadata={"help": "Limit the max number of training examples."}
        # ToDo: impliment
    )


@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: str = field(
        default=f'../data/finetunes/{str(dt.datetime.now())}',
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )            
    optim: str = field(
        default="adamw_torch"
    )
    model_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    bf16: bool = field(
        default=False,
        metadata={"help": "Whether to use bf16 16-bit (mixed) precision training instead of 32-bit training. Requires Ampere or higher NVIDIA architecture"}, 
    )
    fp16: bool = field(
        default=True,
        metadata={"help": "Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training"},
    )
    local_rank: int = field(default=0) # for DDP
    learning_rate: float = field(default=3e-4)
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 4,
    warmup_ratio: float = 0.05,
    num_train_epochs: int = 3,
    logging_steps: int = 200,
    evaluation_strategy: str = "steps",
    save_strategy: str = "steps",
    eval_steps: int = 200
    save_steps: int = 200,
    save_total_limit: int = 3,
    load_best_model_at_end: bool = True,
    ddp_find_unused_parameters: bool = False,

    
def train():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}  # this is for DDP to use 1 GPU per process
    )
    model = prepare_model_for_int8_training(model)

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    
    model.print_trainable_parameters()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.base_model,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        add_eos_token=True
    )
    tokenizer.pad_token = tokenizer.unk_token

    # make the dataset for trainer
    dataset = load_dataset(
        "json", 
        data_files=data_args.data_path, 
        split='train[:]'
    ).train_test_split(test_size=0.05)
    
    def preprocess_dataset(x):
        # ToDo: preprocess the dataset and replace special tokens with unknown tokens
        # ToDo: abstract, and make templates usable
        prompt = str(
            '<|SYSTEM|>You are an experienced financial analyst. You are tasked with responding to user inputs with insightful and helpful replies. User inputs are sent as JSON, you will respond with just text.<|END_SYSTEM|>'
            f'<|USER_INPUT|>{"{"}"title": "{x["title"]}", "input": "{x["selftext"]}"{"}"}<|END_USER_INPUT|>'
            f'<|RESPONSE|>{x["body"]}<|END_RESPONSE|>'
        )
        return {
            **x,
            'prompt': prompt
        }

    prompt_ds = dataset.map(preprocess_dataset)
    ds = prompt_ds.map(
        lambda samples: tokenizer(
            samples['prompt'], 
            truncation=True, 
            max_length=training_args.model_max_length, 
            padding='max_length'
        ), 
        batched=True
    )
    
    trainer = Trainer(
        model=model, 
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        args=training_args,        
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!# silence the warnings. Please re-enable for inference!

    trainer.train()
    trainer.save_state()

    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()