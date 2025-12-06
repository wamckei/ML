# Training Iteration 3 - Resume from previous LoRA model
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:32"
import torch
torch.cuda.empty_cache()
import gc
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, 
    DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from peft import PeftModel, LoraConfig, get_peft_model, TaskType

# Same base model and quantization
model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
previous_model_path = "./my-python-code-llm-final_qwen"

# PLACEHOLDER: Replace with your new/expanded dataset(s)
# Example: new_dataset = load_dataset("your/new-python-dataset", split="train[:5000]")
# dataset = concatenate_datasets([load_dataset("flytech/python-codes-25k", split="train[1000:]"), new_dataset])
dataset = load_dataset("flytech/python-codes-25k", split="train[1000:2000]")  # Placeholder

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Same quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
)

# Load BASE model (quantized)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config,
    device_map="auto", offload_folder="offload",
    trust_remote_code=True, torch_dtype=torch.float16
)

# CRITICAL: Load your PREVIOUS LoRA adapters on top
model = PeftModel.from_pretrained(base_model, previous_model_path)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

# OPTIONAL: Add NEW LoRA layers or adjust existing (increase capacity)
# model = get_peft_model(model, lora_config)  # Only if adding fresh LoRA

model.print_trainable_parameters()

# Same tokenization function
def tokenize_function(examples):
    texts = [f"### Instruction: {inst}\n### Input: {inp}\n### Response: {out}" 
             for inst, inp, out in zip(examples['instruction'], examples['input'], examples['output'])]
    tokenized = tokenizer(texts, truncation=True, padding=False, max_length=512, return_tensors=None)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# Conservative resume training args (LOWER LR for continued training)
training_args = TrainingArguments(
    output_dir="./qwen-coder-finetuned-v2",
    num_train_epochs=2, per_device_train_batch_size=1,
    gradient_accumulation_steps=8, optim="adamw_torch",
    learning_rate=5e-5,  # LOWER for continued training
    fp16=True, logging_steps=25, max_steps=3000,
    save_steps=750, eval_strategy="no",
    warmup_steps=25, dataloader_pin_memory=False,
    gradient_checkpointing=True, remove_unused_columns=False,
    report_to=None, dataloader_num_workers=0,
    group_by_length=True, max_grad_norm=0.3,
    dataloader_drop_last=True, load_best_model_at_end=False
)

gc.collect()
torch.cuda.empty_cache()

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, padding=True)

print("🚀 Resuming training from previous LoRA adapters...")
trainer = Trainer(
    model=model, args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer, data_collator=data_collator
)

trainer.train()
trainer.save_model("./my-python-code-llm-v2_final")
print("✅ Iteration 3 complete!")
