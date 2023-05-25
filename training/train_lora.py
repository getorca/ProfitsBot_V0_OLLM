import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
import transformers
from transformers import Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset


@dataclass
class LoraArguments:
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_weight_path: str = ""
    bias: str = "none"
    

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="huggyllama/llama-7b")


@dataclass
class DataArguments:
    data_path: str = field(
        default='../data/top.jsonl', 
        metadata={"help": "Path to the training data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    local_rank: int = field(default=0) # for DDP
    learning_rate: float = field(default=3e-4)

    
def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}  # this is for DDP to use 1 GPU
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
    
    # if training_args.deepspeed is not None and training_args.local_rank == 0:
    model.print_trainable_parameters()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        add_eos_token=True
    )
    tokenizer.pad_token = tokenizer.unk_token

    # make the dataset for trainer
    dataset = load_dataset("json", data_files=data_args.data_path, split='train[:250000]').train_test_split(test_size=0.1)
    
    def preprocess_dataset(x):
        prompt = str(
                '<|SYSTEM|>\nYou are an experienced financial analyst. You are tasked with responding to user inputs with insightful and helpful replies. User inputs are sent as JSON, you will respond with just text.\n<|END_SYSTEM|>\n'
                f'<|USER_INPUT|>\n'
                f'{"{"}"title": {x["title"]}, "input": {x["selftext"]}{"}"}\n' 
                '<|END_USER_INPUT|>\n'
                f'<|RESPONSE|>\n{x["body"]}\n<|END_RESPONSE|>\n'
        )
        return {
            **x,
            'prompt': prompt
        }

    prompt_ds = dataset.map(preprocess_dataset)
    ds = prompt_ds.map(lambda samples: tokenizer(samples['prompt'], truncation=True, max_length=1024, padding='max_length'), batched=True)

    
    trainer = Trainer(
        model=model, 
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            warmup_steps=300,
            num_train_epochs=3,
            learning_rate=2.5e-4,
            lr_scheduler_type='polynomial', 
            fp16=True,
            logging_steps=100,
            optim='adamw_bnb_8bit', # "adamw_torch"
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=500, # 200
            save_steps=500, # 200
            save_total_limit=3,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False,
            output_dir=training_args.output_dir,
            report_to="wandb"
        ),        
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!# silence the warnings. Please re-enable for inference!

    trainer.train()
    trainer.save_state()

    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()