# utils.py
import importlib
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2TokenizerFast


def get_device():
    """
    Determines the best available device (CUDA, MPS, or CPU) for running the model.
    Returns:
        device: The best available device.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device


def initialize_model_and_tokenizer(model_name, tok=None, mod=None, cache_dir=None):
    """
    Initializes the model and tokenizer based on the model name.
    Args:
        model_name (str): The name of the model to load from Hugging Face.
        tok (str): The name of the tokenizer class to load.
        mod (str): The name of the model class to load.
        cache_dir (str): The directory to cache the model and tokenizer.
    Returns:
        model: The pre-trained language model.
        tokenizer: The tokenizer for the model.
    """
    device = get_device()
    module_name = "transformers"
    module = importlib.import_module(module_name)

    # Initialize the tokenizer
    try:
        if tok:
            tokenizer_class = getattr(module, tok)
            tokenizer = tokenizer_class.from_pretrained(model_name, cache_dir=cache_dir)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    except Exception as e:
        print(f"AutoTokenizer failed with error: {e}. Trying GPT2TokenizerFast.")
        tokenizer = GPT2TokenizerFast.from_pretrained(model_name, cache_dir=cache_dir)

    # Initialize the model
    try:
        if mod:
            model_class = getattr(module, mod)
            model = model_class.from_pretrained(model_name, cache_dir=cache_dir).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir).to(device)
    except Exception as e:
        print(f"Model initialization failed with error: {e}.")
        raise

    return model, tokenizer


def save_model_local(model, tokenizer, output_dir="./output"):
    """
    Saves the model and tokenizer locally.
    Args:
        model: The pre-trained language model.
        tokenizer: The tokenizer for the model.
        output_dir (str): The directory to save the model and tokenizer.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")


def load_model_local(output_dir="./output"):
    """
    Loads the model and tokenizer from the local directory.
    Args:
        output_dir (str): The directory to load the model and tokenizer from.
    Returns:
        model: The pre-trained language model.
        tokenizer: The tokenizer for the model.
    """
    device = get_device()
    model = AutoModelForCausalLM.from_pretrained(output_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    return model, tokenizer
