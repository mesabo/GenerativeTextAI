# main.py
import os
import argparse
import pandas as pd
from utils import get_device, initialize_model_and_tokenizer, save_model_local, load_model_local
from fine_tune import fine_tune_model, save_model
from gpt2_simple_utils import download_gpt2_model, fine_tune_gpt2, generate_gpt2, save_gpt2_model, load_gpt2_model
from datasets import load_dataset

# Set environment variable to avoid OpenMP runtime issue and optimize MPS memory usage
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


def save_results_to_csv(results, model_name, output_dir='./output'):
    """
    Saves the generated text results to a CSV file.
    Args:
        results (list): List of generated text results.
        model_name (str): The name of the model.
        output_dir (str): Directory to save the CSV file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"{model_name}_results.csv")
    df = pd.DataFrame(results, columns=["Generated Text"])
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def main(args):
    instruction = "Given the candidate's details, predict the placement status."
    template = "Based on the following details: CGPA: {CGPA}; Internships: {Internships}; Projects: {Projects}; Workshops/Certifications: {Workshops_Certifications}; Aptitude Test Score: {AptitudeTestScore}; Soft Skills Rating: {SoftSkillsRating}; Extracurricular Activities: {ExtracurricularActivities}; Placement Training: {PlacementTraining}; SSC Marks: {SSC_Marks}; HSC Marks: {HSC_Marks}, predict the placement status."

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
            "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "tok": None,
            "mod": None
        },
        "6": {
            "name": "rasyosef/gpt2-mini-amharic-28k",
            "tok": None,
            "mod": None
        },
        "7": {
            "name": "gpt2-simple",
            "tok": None,
            "mod": "gpt-2-simple"
        }
    }

    selected_model_key = args.model_key
    selected_model = my_model[selected_model_key]
    model_name = selected_model["name"]

    dataset = load_dataset("Krooz/Campus_Recruitment_CSV")
    df = pd.DataFrame(dataset['train'])

    if selected_model_key == "7":
        import gpt_2_simple as gpt2
        model_name = "124M"
        run_name = "run1"
        checkpoint_dir = os.path.join("checkpoint", run_name)

        download_gpt2_model(model_name=model_name)

        if args.action == 'fine_tune':
            sess = fine_tune_gpt2(df, model_name=model_name, steps=1000)
            save_gpt2_model(sess, run_name=run_name)
        else:
            sess = gpt2.start_tf_sess()
            load_gpt2_model(sess, run_name=run_name)

        generated_texts = generate_gpt2(sess, return_as_list=True)
        save_results_to_csv(generated_texts, model_name)
    else:
        model_dir = f'./output/{model_name.replace("/", "_")}'

        if os.path.exists(model_dir) and args.action == 'generate':
            model, tokenizer = load_model_local(model_dir)
        else:
            model, tokenizer = initialize_model_and_tokenizer(selected_model["name"], selected_model["tok"],
                                                              selected_model["mod"], cache_dir="./model_cache")
            save_model_local(model, tokenizer, model_dir)

        if args.action == 'fine_tune':
            trainer = fine_tune_model(model, tokenizer, epochs=args.epochs)
            save_model(trainer)
        else:
            generated_texts = []
            for i, row in df.iterrows():
                input_text = template.format(
                    CGPA=row['CGPA'],
                    Internships=row['Internships'],
                    Projects=row['Projects'],
                    Workshops_Certifications=row['Workshops/Certifications'],
                    AptitudeTestScore=row['AptitudeTestScore'],
                    SoftSkillsRating=row['SoftSkillsRating'],
                    ExtracurricularActivities=row['ExtracurricularActivities'],
                    PlacementTraining=row['PlacementTraining'],
                    SSC_Marks=row['SSC_Marks'],
                    HSC_Marks=row['HSC_Marks']
                )
                prompt = f"{instruction}\n{input_text}"
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
                output = model.generate(input_ids, max_length=100)
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                generated_texts.append(generated_text)

            save_results_to_csv(generated_texts, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune or Generate using a specified model")
    parser.add_argument('action', choices=['fine_tune', 'generate'],
                        help="Action to perform: 'fine_tune' or 'generate'")
    parser.add_argument('--model_key', type=str, default='5', help="Key of the model to use from the model dictionary")
    parser.add_argument('--epochs', type=int, default=3, help="Number of epochs for fine-tuning")

    args = parser.parse_args()
    main(args)
