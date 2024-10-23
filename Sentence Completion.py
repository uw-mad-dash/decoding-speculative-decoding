# Install necessary libraries (if not already installed)
!pip install transformers
import torch
import torch.nn as nn
import re
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch.distributed as dist
import json
import os
# Import necessary modules
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Select the model and dataset
model_name = "facebook/opt-2.7b"  # Change to any other model from Hugging Face

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()  # Set model to evaluation mode

def json_loader(file_path):
    """
    Input: file_path: Path to the json file containing all the queries.

    File format looks like the following:
    {"prompt": "A seated man cleans a shoe in a classroom setting with other individuals. the man"}
    {"prompt": "Two girls are sailing on a lake. they"}

    Output: This function returns a list of prompts to be used by the draft LLM.
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

hellaswag_test = json_loader("/content/hellaswag.json")

# Set the output file name
output_file = "generated_completions.txt"

# Open the file in write mode
with open(output_file, "w", encoding="utf-8") as f:
    # Evaluate on a subset for demonstration
    num_examples = 200  # Change this number to generate more examples

    for i in range(num_examples):
        example = hellaswag_test[i]
        context = example['prompt'].strip()

        # Encode the context
        inputs = tokenizer.encode(context, return_tensors='pt').to(device)

        # Generate continuation
        max_length = inputs.shape[1] + 200  # Adjust the max_length as needed
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Write only the tensor values without device info
        tensor_values = outputs[:, inputs.size(1):].cpu().numpy()  # Move tensor to CPU and convert to numpy
        f.write(f"tensor({tensor_values.tolist()})\n")  # Convert numpy array to list format for writing

        print(f"Example {i+1} completed and written to file.")

print(f"All completions have been written to {output_file}")

