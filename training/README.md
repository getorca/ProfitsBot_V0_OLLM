# Training

## Setup

### 1st - a cuda environment

- the easiest way to get this is with conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html>
- `conda create -n trainer python=3.10`
- `conda activate trainer`
- `conda install -c conda-forge cudatoolkit-dev`

### 2nd - install the requirements

- `pip install -r requirements.txt`

This uses some older versions of transformers and bitsandbytes, becasue they are tested to be working. There are some updates in the works for those repos, but I have yet to test them.

## Running the training

This uses DDP for parallesim.

So for example to run it on a 5 gpu machine you would run:

```bash
OMP_NUM_THREADS=4 WORLD_SIZE=5 torchrun --nproc_per_node=5 --master_port=1234 train_lora.py \
    --output_dir '/data/tunned_2' \
    --model_name_or_path '/models/my_llama_hf/llama_hf_7B'
```

on a single gpu you would run:

```bash
OMP_NUM_THREADS=4 WORLD_SIZE=1 torchrun --nproc_per_node=1 --master_port=1234 train_lora.py \
    --output_dir '/data/tunned_2' \
    --model_name_or_path '/models/my_llama_hf/llama_hf_7B'
```

deepspeed or accelerate otions may come. 