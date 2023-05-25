# ProfitsBot_V0_OLLM

ProfitsBot V0 are a set of LLM experiments training open source langage models with loras for financial applications.

## The Models

| Model | Base Model | Dataset | Description | Training |
|-------|------------|---------| -------- | ---- |
| [pb_lora_7b_v0.1](https://huggingface.co/winddude/pb_lora_7b_v0.1) | llama 7b | [reddit_finance_43_250k](https://huggingface.co/datasets/winddude/reddit_finance_43_250k) | An experimental model trained to reply to various finance, crypto and investing subreddits | [Training](https://github.com/getorca/ProfitsBot_V0_OLLM/blob/main/training) |

## The Datasets

| Dataset | Size | Source | Description | Recreation |
|---------|------|--------|-------------|------------|
| [reddit_finance_43_250k](https://huggingface.co/datasets/winddude/reddit_finance_43_250k) | 250k | reddit | currated top post/comment pairs built from 43 finance subreddits | [Building](https://github.com/getorca/ProfitsBot_V0_OLLM/blob/main/ds_builder) |
