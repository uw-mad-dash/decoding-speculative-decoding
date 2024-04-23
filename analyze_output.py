import torch
import re

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

def compute_average_difference(tensors):
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
    tensor_stack = torch.stack(tensors)
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

    return avg_tokens_matched_per_prompt, avg_tokens_matched


# Path to your text file
output_file_path = '/path/to/your/output/file'

# Read the tensors from the file
tensors = read_tensors_from_file(output_file_path)

# Compute the average differences
avg_prompt, avg_dataset = compute_average_difference(tensors)

# Print the result
print("Average tokens matched per prompt:", avg_prompt)
print("Average tokens matched per dataset:", avg_dataset)
