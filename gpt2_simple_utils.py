#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/28/2024
🚀 Welcome to the Awesome Python Script 🚀

User: messou
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

# gpt2_simple_utils.py
import os

import gpt_2_simple as gpt2
import pandas as pd
from sklearn.model_selection import train_test_split


def download_gpt2_model(model_name="124M"):
    """
    Downloads the GPT-2 model if it's not already present.
    Args:
        model_name (str): The name of the GPT-2 model to download.
    """
    if not os.path.isdir(os.path.join("models", model_name)):
        print(f"Downloading {model_name} model...")
        gpt2.download_gpt2(model_name=model_name)  # model is saved into current directory under /models/124M/


def fine_tune_gpt2(dataframe, model_name="124M", steps=1000):
    """
    Fine-tunes the GPT-2 model using the provided dataset.
    Args:
        dataframe (pd.DataFrame): The dataframe containing the training data.
        model_name (str): The name of the GPT-2 model to fine-tune.
        steps (int): The number of training steps.
    """
    texts = dataframe.apply(lambda
                                row: f"CGPA: {row['CGPA']}; Internships: {row['Internships']}; Projects: {row['Projects']}; Workshops/Certifications: {row['Workshops/Certifications']}; Aptitude Test Score: {row['AptitudeTestScore']}; Soft Skills Rating: {row['SoftSkillsRating']}; Extracurricular Activities: {row['ExtracurricularActivities']}; Placement Training: {row['PlacementTraining']}; SSC Marks: {row['SSC_Marks']}; HSC Marks: {row['HSC_Marks']}",
                            axis=1).tolist()

    train_texts, val_texts = train_test_split(texts, test_size=0.1)

    train_file = "train.txt"
    with open(train_file, 'w') as f:
        for text in train_texts:
            f.write(text + "\n")

    sess = gpt2.start_tf_sess()
    gpt2.finetune(sess,
                  train_file,
                  model_name=model_name,
                  steps=steps)  # steps is max number of training steps
    return sess


def generate_gpt2(sess, return_as_list=False):
    """
    Generates text using the fine-tuned GPT-2 model.
    Args:
        sess: The TensorFlow session with the fine-tuned model.
        return_as_list (bool): Whether to return the generated text as a list.
    """
    return gpt2.generate(sess, return_as_list=return_as_list)


def save_gpt2_model(sess, run_name="run1"):
    """
    Saves the fine-tuned GPT-2 model.
    Args:
        sess: The TensorFlow session with the fine-tuned model.
        run_name (str): The name of the run for saving the model.
    """
    gpt2.save_gpt2(sess, run_name=run_name)


def load_gpt2_model(sess, run_name="run1"):
    """
    Loads the fine-tuned GPT-2 model.
    Args:
        sess: The TensorFlow session with the fine-tuned model.
        run_name (str): The name of the run to load the model from.
    """
    gpt2.load_gpt2(sess, run_name=run_name)
