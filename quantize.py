#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/26/2024
🚀 Welcome to the Awesome Python Script 🚀

User: messou
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""
# quantize.py
import torch

def quantize_model(model):
    """
    Quantizes the model using dynamic quantization.
    Args:
        model: The pre-trained language model.
    Returns:
        quantized_model: The quantized model.
    """
    # Ensure the quantization engine is set to qnnpack for macOS
    torch.backends.quantized.engine = 'qnnpack'

    # Apply dynamic quantization (must be done on CPU)
    model.to('cpu')
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model
