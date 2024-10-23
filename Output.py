import torch
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Function to read output from the output file and convert to a tensor
def read_tensors_from_file(file_path):
    tensors = []
    with open(file_path, 'r') as file:
        for line in file:
            # Extract all numbers from brackets
            matches = re.findall(r'\[(.*?)\]', line)
            if matches:
                # Handle multiple comma-separated numbers in the brackets
                for match in matches:
                    numbers = match.split(',')
                    # Convert each number from string to integer
                    tensor = torch.tensor([int(num.strip()) for num in numbers if num.strip().isdigit()], dtype=torch.int32)
                    tensors.append(tensor)
    return tensors




def compute_average_difference(tensors,confidence_level = 0.95):
    """
    Computes the average difference between consecutive tensors in a list.

    Parameters:
    tensors (list of torch.Tensor): A list of tensors for which the average differences are calculated.

    Returns:
    tuple: A tuple containing three elements:
        - avg_tokens_matched_per_prompt (list of torch.Tensor): A list of average differences computed segment-wise,
          when the difference changes the sign to negative.
        - avg_tokens_matched (torch.Tensor): The overall average difference across all tensors.

    """
    max_size = max(tensor.shape[0] for tensor in tensors)
    padded_tensors = []
    for tensor in tensors:
        padded_tensor = torch.zeros(max_size, dtype=torch.float32)
        padded_tensor[:tensor.shape[0]] = tensor
        padded_tensors.append(padded_tensor)
    tensor_stack = torch.stack(padded_tensors)

    #tensor_stack = torch.stack(tensors)
    differences = torch.zeros_like(tensor_stack)
    segment_start = 0
    avg_tokens_matched_per_prompt = []

    for i in range(1, len(tensor_stack)):
        diff = tensor_stack[i] - tensor_stack[i-1]
        if diff[0] < 0:
            new_diff = differences[segment_start:i-1]
            avg_tokens_matched_per_prompt.append(torch.mean(new_diff.float(), dim=0))
            segment_start = i

        differences[i-1] = torch.where(diff > 0, diff, tensor_stack[i])

    # Handle the last segment
    new_diff = differences[segment_start:]
    avg_tokens_matched_per_prompt.append(torch.mean(new_diff.float(), dim=0))
    differences[-1] = torch.where(diff > 0, diff, tensor_stack[-1])
    # Compute overall average of differences
    avg_tokens_matched = torch.mean(differences.float(), dim=0)

    # Compute standard deviation of differences
    std_tokens_matched = torch.std(differences.float(), dim=0)

    # Compute confidence interval
    n = differences.shape[0]
    se = std_tokens_matched / torch.sqrt(torch.tensor(n, dtype=torch.float32))
    h = se * torch.tensor(stats.t.ppf((1 + confidence_level) / 2., n - 1), dtype=torch.float32)
    confidence_interval = (avg_tokens_matched - h, avg_tokens_matched + h)

    return avg_tokens_matched_per_prompt, avg_tokens_matched, std_tokens_matched, confidence_interval




# Path to your text file
output_file_path = '/content/llama-796m-hellaswag-lookahead-2.txt'

# Read the tensors from the file
tensors = read_tensors_from_file(output_file_path)

# Compute the average differences
avg_prompt, avg_dataset, std_tokens_matched, confidence_interval = compute_average_difference(tensors)

# Print the result
print("Average tokens matched per prompt:", avg_prompt)
print("Average tokens matched per dataset:", avg_dataset)

num_tensors = len(avg_prompt)

print(f"The number of prompts in the list is: {num_tensors}")

print("Standard Deviation:",std_tokens_matched)
print("confidence interval:",confidence_interval)


