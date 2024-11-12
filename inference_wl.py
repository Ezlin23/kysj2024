from unsloth import FastLanguageModel
import torch
max_seq_length = 16384 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    #model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    model_name = "lora_model",
    #model_name = "/home/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
tokenizer.pad_token = tokenizer.eos_token

import json


with open("messages_120_test.json", "r") as file:
    messages = json.load(file)
print(messages[0])

input_text = [tokenizer.apply_chat_template(message, tokenize=False) for message in messages]
print(input_text[0])

from datasets import Dataset

tokenized_dataset = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
# Convert tokenized dataset to Hugging Face Dataset
dataset = Dataset.from_dict(tokenized_dataset)

with open("messages_120_test.json", "r") as file:
    filtered_messages = json.load(file)

filtered_input_text = [tokenizer.apply_chat_template(message, tokenize=False) for message in filtered_messages]
print(filtered_input_text[0])

# %%
# alpaca_prompt = Copied from above
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

inference_inputs_t = [tokenizer(input_text, return_tensors = "pt").to("cuda") for input_text in filtered_input_text]

inference_outputs_t_wnl = [model.generate(**inputs, max_new_tokens = 16384, use_cache = True) for inputs in inference_inputs_t]
with open("inference_outputs_wl_120.json", "w") as file:
    json.dump([tokenizer.decode(output[0], skip_special_tokens=True) for output in inference_outputs_t_wnl], file)

