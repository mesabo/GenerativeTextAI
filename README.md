# Project structure

project_root/
│
├── main.py
├── fine_tune.py
├── utils.py
├── gpt2_simple_utils.py
├── data/ # Directory for datasets
├── output/ # Directory for output models and CSV files
└── logs/ # Directory for logs

# Run the code

- To train a model:

> python main.py fine_tune --model_key 7 --epochs 3

- To predict using an existing model:

> python main.py generate --model_key 7

# GenerativeTextAI
