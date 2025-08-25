# -*- coding: utf-8 -*-
"""RelExtract-Unsloth_copy0.ipynb

"""

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# import os
# if "COLAB_" not in "".join(os.environ.keys()):
#     !pip install unsloth
# else:
#     # Do this only in Colab notebooks! Otherwise use pip install unsloth
#     !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
#     !pip install sentencepiece protobuf "datasets>=3.4.1" huggingface_hub hf_transfer
#     !pip install --no-deps unsloth

from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 15 trillion tokens model 2x faster!
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # We also uploaded 4bit for 405b!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # New Mistral 12b 2x faster!
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

import json

with open('/content/drive/MyDrive/radgraph/train.json', 'r') as file:
    data = json.load(file)

import json

with open('/content/drive/MyDrive/radgraph/dev.json', 'r') as file:
    eval = json.load(file)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

import pandas as pd
import csv


def rewrite_values(code):
    mapping = {
        "ANAT-DP": "Anatomy, Definitely Present",
        "OBS-DP": "Observation, Definitely Present",
        "OBS-U": "Observation, Uncertain",
        "OBS-DA": "Observation, Definitely Absent"
    }
    return mapping.get(code, "Unknown code")

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""


def build_input(ent1, label1, ent2, label2, note):
    return (
        "You are an expert in clinical natural language processing.\n"
        "Based on the information provided, determine the relationship between two entities in a radiology sentence.\n\n"
        "Here are the definitions of the entities and possible relationships:\n\n"
        "Anatomy: Refers to anatomical body parts found in radiology reports, such as \"lung\".\n"
        "Observation: Refers to findings or diagnoses from a radiology image, such as \"effusion\", \"increased\", or \"clear\".\n"
        "Possible relationships:\n\n"
        "suggestive_of (Observation, Observation): One observation implies or suggests another observation.\n"
        "located_at (Observation, Anatomy): An observation is located at or associated with an anatomical body part.\n"
        "modify (Observation, Observation) or (Anatomy, Anatomy): One entity modifies or quantifies the other.\n"
        "Now classify the relation between the following two entities found in the same sentence:\n\n"
        f'What is the correct relationship between "{ent1}" ({label1}) and "{ent2}" ({label2}), based on their use in the same radiology sentence below?\n'
        f'"{note}"\n\n'
        "Choose only one of the following relations: suggestive_of, located_at, modify\n"
        "Just output the chosen relation as a single word (no explanation)"
    )


data_list = []
for fileid, ld in data.items():
    ent = ld['entities']
    for eid, ent1 in ent.items():
        for rel in ent1.get("relations", []):
            rel_type, eid2 = rel
            ent2 = ent[eid2]
            data_list.append({
                "instruction": "Classify the relationship between two entities in a radiology sentence.",
                "input": build_input(ent1['tokens'], ent1['label'], ent2['tokens'], ent2['label'], ld['text']),
                "output": rel_type
            })

df = pd.DataFrame(data_list)
df.to_json("rel_extract_alpaca.json", orient="records", lines=True)
print(df.head(1))
from datasets import load_dataset

dataset = load_dataset("json", data_files="rel_extract_alpaca.json", split="train")

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(
            instruction=instruction,
            input=input,
            output=output
        ) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}
dataset = dataset.map(formatting_prompts_func, batched=True)

from trl import SFTConfig, SFTTrainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs=4,
        max_steps = 200,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
        ),
    )

trainer_stats = trainer.train()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logs = trainer.state.log_history

rows = []
for r in logs:
    if "loss" in r and "epoch" in r and "eval_loss" not in r:
        e = int(np.floor(r["epoch"])) + 1
        rows.append({"epoch": e, "loss": r["loss"]})

df = pd.DataFrame(rows).groupby("epoch", as_index=False)["loss"].mean().sort_values("epoch")

plt.figure(figsize=(7,4))
plt.plot(df["epoch"], df["loss"], marker="o", label="Train Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training Loss Across Epochs")
plt.xticks(df["epoch"])
plt.grid(alpha=0.2); plt.legend(); plt.show()

"""Validation"""

import re


golden_tag_list = []
predicted_tag_list = []

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Classify the relationship between two entities in a radiology sentence.

### Input:
You are an expert in clinical natural language processing.
Based on the information provided, determine the relationship between two entities in a radiology sentence.

Here are the definitions of the entities and possible relationships:

Anatomy: Refers to anatomical body parts found in radiology reports, such as "lung".
Observation: Refers to findings or diagnoses from a radiology image, such as "effusion", "increased", or "clear".
Possible relationships:

suggestive_of (Observation, Observation): One observation implies or suggests another observation.
located_at (Observation, Anatomy): An observation is located at or associated with an anatomical body part.
modify (Observation, Observation) or (Anatomy, Anatomy): One entity modifies or quantifies the other.
Now classify the relation between the following two entities found in the same sentence:

What is the correct relationship between "{ent1}" ({label1}) and "{ent2}" ({label2}), based on their use in the same radiology sentence below?
"{note}"

Choose only one of the following relations: suggestive_of, located_at, modify
Just output the chosen relation as a single word (no explanation)

### Response:
"""

for i, d in enumerate(eval):
    if i > 10:
        break
    ld = eval[d]

    relations = [(e, ent[e]['relations']) for e in ent.keys()]

    final_rels = []
    for rel in relations:
        rel_init = ent[rel[0]]
        for local_rel in rel[1]:
            rel_2nd = ent[local_rel[1]]

            golden_tag_list.append(local_rel[0])

            prompt = alpaca_prompt.format(
                ent1=rel_init['tokens'],
                label1=rewrite_values(rel_init['label']),
                ent2=rel_2nd['tokens'],
                label2=rewrite_values(rel_2nd['label']),
                note=ld['text']
            )

            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=16, use_cache=True)
            response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            last_word = response.strip().split()[-1].strip(".,:;\"'").lower()

            predicted_tag_list.append(last_word)
            print("GOLD:", local_rel[0], "PRED:", last_word)
            print("-" * 20)


print("Gold:", golden_tag_list)
print("Predicted:", predicted_tag_list)

from sklearn.metrics import f1_score, classification_report

labels = ["suggestive_of", "located_at", "modify"]

micro_f1 = f1_score(golden_tag_list, predicted_tag_list, average='micro', labels=labels, zero_division=0)
macro_f1 = f1_score(golden_tag_list, predicted_tag_list, average='macro', labels=labels, zero_division=0)

print(f"Micro F1 Score: {micro_f1:.4f}")
print(f"Macro F1 Score: {macro_f1:.4f}")

labels = sorted(set(golden_tag_list + predicted_tag_list))

f1 = f1_score(golden_tag_list, predicted_tag_list, labels=labels, average=None)

for label, score in zip(labels, f1):
    print(f"Label: {label}, F1 Score: {score:.4f}")

"""Evaluation"""

import json

with open('/content/drive/MyDrive/radgraph/test.json', 'r') as file:
    test_set = json.load(file)

import re


golden_tag_list = []
predicted_tag_list = []

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Classify the relationship between two entities in a radiology sentence.

### Input:
You are an expert in clinical natural language processing.
Based on the information provided, determine the relationship between two entities in a radiology sentence.

Here are the definitions of the entities and possible relationships:

Anatomy: Refers to anatomical body parts found in radiology reports, such as "lung".
Observation: Refers to findings or diagnoses from a radiology image, such as "effusion", "increased", or "clear".
Possible relationships:

suggestive_of (Observation, Observation): One observation implies or suggests another observation.
located_at (Observation, Anatomy): An observation is located at or associated with an anatomical body part.
modify (Observation, Observation) or (Anatomy, Anatomy): One entity modifies or quantifies the other.
Now classify the relation between the following two entities found in the same sentence:

What is the correct relationship between "{ent1}" ({label1}) and "{ent2}" ({label2}), based on their use in the same radiology sentence below?
"{note}"

Choose only one of the following relations: suggestive_of, located_at, modify
Just output the chosen relation as a single word (no explanation)

### Response:
"""

for i, d in enumerate(data):
    if i > 10:
        break
    ld = data[d]
    ent = ld['entities']

    relations = [(e, ent[e]['relations']) for e in ent.keys()]

    final_rels = []
    for rel in relations:
        rel_init = ent[rel[0]]
        for local_rel in rel[1]:
            rel_2nd = ent[local_rel[1]]

            golden_tag_list.append(local_rel[0])

            prompt = alpaca_prompt.format(
                ent1=rel_init['tokens'],
                label1=rewrite_values(rel_init['label']),
                ent2=rel_2nd['tokens'],
                label2=rewrite_values(rel_2nd['label']),
                note=ld['text']
            )

            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=16, use_cache=True)
            response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            last_word = response.strip().split()[-1].strip(".,:;\"'").lower()

            predicted_tag_list.append(last_word)
            print("GOLD:", local_rel[0], "PRED:", last_word)
            print("-" * 20)


print("Gold:", golden_tag_list)
print("Predicted:", predicted_tag_list)

from sklearn.metrics import f1_score, classification_report

labels = ["suggestive_of", "located_at", "modify"]

micro_f1 = f1_score(golden_tag_list, predicted_tag_list, average='micro', labels=labels, zero_division=0)
macro_f1 = f1_score(golden_tag_list, predicted_tag_list, average='macro', labels=labels, zero_division=0)

print(f"Micro F1 Score: {micro_f1:.4f}")
print(f"Macro F1 Score: {macro_f1:.4f}")

labels = sorted(set(golden_tag_list + predicted_tag_list))

f1 = f1_score(golden_tag_list, predicted_tag_list, labels=labels, average=None)

for label, score in zip(labels, f1):
    print(f"Label: {label}, F1 Score: {score:.4f}")
