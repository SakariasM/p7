#!/bin/bash
python3 -m venv ~/llm_venv
source ~/llm_venv/bin/activate
pip install --upgrade pip
pip install vllm transformers peft accelerate bitsandbytes huggingface_hub
