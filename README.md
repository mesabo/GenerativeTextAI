# Project structure

.
├── README.md
├── checkpoint
├── data
├── fine_tune.py
├── gpt2_simple_utils.py
├── logs
├── main.py
├── model_cache
├── models
├── notebooks
├── output
├── quantize.py
├── requirements.txt
├── shakespeare.txt
└── utils.py

# Run the code

- To train a model:

> python main.py fine_tune --model_key 7 --epochs 3

- To predict using an existing model:

> python main.py generate --model_key 7

# GenerativeTextAI
