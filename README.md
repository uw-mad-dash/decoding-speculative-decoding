# Decoding Speculative Decoding
[Arxiv Preprint](https://arxiv.org/pdf/2402.01528.pdf)

Distilled draft models: [1.3B](https://huggingface.co/minghaoyan/Wide-Sheared-LLaMA-1.3B) | [796M](https://huggingface.co/minghaoyan/Wide-Sheared-LLaMA-796M) | [543M](https://huggingface.co/minghaoyan/Wide-Sheared-LLaMA-543M) | [290M](https://huggingface.co/minghaoyan/Wide-Sheared-LLaMA-290M).

## Introduction
This repo contains two python script demonstrating speculative decoding on LLaMA models. We also release a series of draft models distilled based on the [Sheared-LLaMA](https://github.com/princeton-nlp/LLM-Shearing) codebase. 

The `speculative_decoding_demo` script is for those who can't deploy a large LLM to try speculative decoding. We pre-compute results from LLaMA-65B and store the output in a log file to simulate speculative decoding.

The `speculative_decoding_deployment` script is for those who wish to deploy their own target and draft LLMs. We build the demo on top of DeepSpeed Inference and HuggingFace libraries.

To launch scripts, please run `deepspeed --num_gpus <# GPUs of your choice> <script name>`. For `speculative_decoding_demo`, 1 GPU is sufficient; for `speculative_decoding_deployment`, it depends on how large your target LLM is. For reference, 4 80GB A100 GPUS are required for running `speculative_decoding_deployment` with 70B models.

For those who wish to try our distilled models, please use the links above to download the draft models. As per our paper, LLaMA-796M is the best performing model and is thus selected as the default model in the notebooks.

This project aims to understand the design space of draft models in speculative decoding. Our key observations is that draft model inference latency bottlenecks the througput of speculative decoding. Furthur, draft model depth is the key bottleneck in draft model inference latency. We release a series of models distilled from LLaMA-7B based on [Sheared-LLaMA](https://github.com/princeton-nlp/LLM-Shearing) to create draft models that significantly ourperforms existing models.

## Install requirements
The speculative decoding demo is based on DeepSpeed and HuggingFace Transformers. Please install Python 3.9 and follow the steps below to install relevant packages:
```
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets deepspeed
```

## Deploy speculative decoding
We provide two scripts to help you deploy speculative decoding, one for those who can deploy a large LLM and one for those who cannot afford to deploy large LLM with pre-computed results stored in advance.

To run the notebook, you will need to make the following changes (These steps are also in the comments in the notebook, feel free to jump into the notebooks directly):

### Set up your test datasets
- `test_json`: Replace the json file with your file path. The format of the json file is specified in the `json_loader` function. We provide a json file with prompts from the Hellaswag dataset.
- `input_file_path` (Only for `speculative_decoding_demo`): Replace the input file with your input file. The format of the input file in specified in the `parse_tensors_from_file` function. We provide outputs from LLaMA-65B on Hellaswag datasets with maximum generation length set to be 200 for each prompt.

### Set up Huggingface target and draft models
- `TRANSFORMERS_CACHE`: Update your transformers_cache directory for saving existing models.
- `model_name`: Update model_name with your targe LLM model name.
- `checkpoint_dir`: Update the checkpoint directory where you save your target LLM checkpoints.
- `checkpoint_files`: Update the checkpoint file names in case your files are named differently.
- `tensor_parallel_degrees`: Update the tensor parallel degrees of your target LLMs used for inference.
- `draft_model`: Update draft_model with your draft LLM model.

### Update the LLaMA tokenizer
The LLaMA tokenizer does not have `[PAD]` token. However, to support batch inference, we need to update the LLaMA tokenizer to include the `[PAD]` token. The following snippet updates the tokenizer and the draft model accordingly. It is recommended to do the same thing on your target LLM and save the updated model checkpoints in local storage as you cannot modify the model embedding layer if you wish to use DeepSpeed on-device model loading from model config:
```
# The LLaMA tokenizer does not have a pad token. 
# Modify the tokenizer to add a pad token and change the model configs accordingly.
tokenizer = AutoTokenizer.from_pretrained("/your/model/name", padding_side="left", torch_dtype=torch.float16)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = "[PAD]"
tokenizer.padding_side = "left"
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)


draft_model = AutoModelForCausalLM.from_pretrained("/your/draft/model/name", torch_dtype=torch.float16)

draft_model.resize_token_embeddings(len(tokenizer))
draft_model.config.pad_token_id = tokenizer.pad_token_id
draft_model.embed_tokens = nn.Embedding(draft_model.config.vocab_size, draft_model.config.hidden_size, padding_idx=draft_model.config.pad_token_id)
```

### Set speculative decoding hyperparameters:

- `batch_size`: Inference batchsize.
- `max_new_tokens`: The amount of tokens the draft model generates during each speculative decoding iteration.
- `output_file`: Name of the output file to store the speculative decoding results.

## Contact
If you have questions related to the paper and code, please email [Minghao](myan@cs.wisc.edu). For bug reports, please either email [Minghao](myan@cs.wisc.edu) or open an issue.

## Citation
Please cite our work if you find it useful!

```bibtex
@article{yan2024decoding,
  title={Decoding Speculative Decoding},
  author={Yan, Minghao and Agarwal, Saurabh and Venkataraman, Shivaram},
  journal={arXiv preprint arXiv:2402.01528},
  year={2024}
}
```








