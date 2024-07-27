# main.py
import os

# Set environment variable to avoid OpenMP runtime issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import importlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2TokenizerFast
from quantize import quantize_model
from fine_tune import fine_tune_model, save_model

def get_device():
    """
    Get the available device (cuda, mps, or cpu).
    Returns:
        device: The available device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def initialize_model_and_tokenizer(model_name, tok=None, mod=None):
    """
    Initializes the model and tokenizer based on the model name.
    Args:
        model_name (str): The name of the model to load from Hugging Face.
        tok (str): The name of the tokenizer class to load.
        mod (str): The name of the model class to load.
    Returns:
        model: The pre-trained language model.
        tokenizer: The tokenizer for the model.
    """
    module_name = "transformers"
    module = importlib.import_module(module_name)

    # Initialize the tokenizer
    try:
        if tok:
            tokenizer_class = getattr(module, tok)
            tokenizer = tokenizer_class.from_pretrained(model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"AutoTokenizer failed with error: {e}. Trying GPT2TokenizerFast.")
        tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

    # Initialize the model
    try:
        if mod:
            model_class = getattr(module, mod)
            model = model_class.from_pretrained(model_name)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name)
    except Exception as e:
        print(f"Model initialization failed with error: {e}.")
        raise

    # Quantize the model
    quantized_model = quantize_model(model)

    # Load model onto the appropriate device after quantization
    device = get_device()
    quantized_model.to(device)

    return quantized_model, tokenizer

if __name__ == "__main__":
    # Define your instruction, list of interests, and template
    instruction = "Given the candidate's details, predict the placement status."
    interests = [
        "CGPA: 8.0; Internships: 2; Projects: 3; Workshops/Certifications: 5; Aptitude Test Score: 80; Soft Skills Rating: 7; Extracurricular Activities: 3; Placement Training: Yes; SSC Marks: 85; HSC Marks: 88",
        "CGPA: 7.5; Internships: 1; Projects: 2; Workshops/Certifications: 3; Aptitude Test Score: 75; Soft Skills Rating: 6; Extracurricular Activities: 2; Placement Training: No; SSC Marks: 78; HSC Marks: 82"
    ]
    template = "Based on the following details: CGPA: {CGPA}; Internships: {Internships}; Projects: {Projects}; Workshops/Certifications: {Workshops/Certifications}; Aptitude Test Score: {AptitudeTestScore}; Soft Skills Rating: {SoftSkillsRating}; Extracurricular Activities: {ExtracurricularActivities}; Placement Training: {PlacementTraining}; SSC Marks: {SSC_Marks}; HSC Marks: {HSC_Marks}, predict the placement status."

    # Provide the model name
    my_model = {
        "0": {
            "name": "HuggingFaceTB/SmolLM-135M",
            "tok": "GPT2TokenizerFast",
            "mod": "AutoModelForCausalLM"
        },
        "1": {
            "name": "HuggingFaceTB/SmolLM-1.7B",
            "tok": None,
            "mod": None
        },
        "2": {
            "name": "PleIAs/OCRonos",
            "tok": None,
            "mod": None
        },
        "3": {
            "name": "Xenova/gpt-4o",
            "tok": "GPT2TokenizerFast",
            "mod": "AutoModelForCausalLM"
        },
        "4": {
            "name": "shreyas1104/bert-scienceQA-v1",
            "tok": None,
            "mod": None
        },
        "5": {
            "name": "unsloth/Phi-3-mini-4k-instruct",
            "tok": None,
            "mod": None
        },
        "6": {
            "name": "rasyosef/gpt2-mini-amharic-28k",
            "tok": None,
            "mod": None
        }
    }

    # Select a model from my_model
    selected_model_key = "5"  # Change this to select a different model
    selected_model = my_model[selected_model_key]

    # Initialize model and tokenizer
    model, tokenizer = initialize_model_and_tokenizer(selected_model["name"], selected_model["tok"], selected_model["mod"])

    # Fine-tune the model
    trainer = fine_tune_model(model, tokenizer)

    # Save the fine-tuned model
    save_model(trainer)