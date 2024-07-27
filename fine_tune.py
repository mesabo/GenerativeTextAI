# fine_tune.py
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling


def fine_tune_model(model, tokenizer, dataset_name="Krooz/Campus_Recruitment_CSV", epochs=3):
    """
    Fine-tunes the model using the provided dataset.
    Args:
        model: The pre-trained language model.
        tokenizer: The tokenizer for the model.
        dataset_name (str): The name of the dataset to load.
        epochs (int): The number of training epochs.
    Returns:
        trainer: The trainer used for fine-tuning.
    """
    # Load the dataset
    dataset = load_dataset(dataset_name)
    df = pd.DataFrame(dataset['train'])

    # Preprocess the dataset
    texts = df.apply(lambda
                         row: f"CGPA: {row['CGPA']}; Internships: {row['Internships']}; Projects: {row['Projects']}; Workshops/Certifications: {row['Workshops/Certifications']}; Aptitude Test Score: {row['AptitudeTestScore']}; Soft Skills Rating: {row['SoftSkillsRating']}; Extracurricular Activities: {row['ExtracurricularActivities']}; Placement Training: {row['PlacementTraining']}; SSC Marks: {row['SSC_Marks']}; HSC Marks: {row['HSC_Marks']}",
                     axis=1).tolist()
    labels = df['PlacementStatus'].apply(lambda x: 1 if x == 'Placed' else 0).tolist()  # Convert labels to integers

    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

        def __len__(self):
            return len(self.encodings.input_ids)

    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)

    # Set up the trainer
    training_args = TrainingArguments(
        output_dir='./output',  # output directory
        num_train_epochs=epochs,  # total number of training epochs
        per_device_train_batch_size=2,  # batch size for training
        per_device_eval_batch_size=2,  # batch size for evaluation
        warmup_steps=10,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    return trainer


def save_model(trainer, output_dir="./output"):
    """
    Saves the fine-tuned model to the specified directory.
    Args:
        trainer: The trainer used for fine-tuning.
        output_dir (str): The directory to save the model.
    """
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    trainer.save_model(output_dir)
