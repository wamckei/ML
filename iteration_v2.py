#!/usr/bin/env python3
"""
Test Your Trained Python Code SLM - Simple & Clean
Requires: ./my-python-code-llm-final/ folder from training
"""

import os
import re
import torch
import sys
from pathlib import Path

# GPU setup
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:32"
torch.cuda.empty_cache()

# Imports
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_DIR = "./my-python-code-llm-final"
BASE_MODEL = "microsoft/DialoGPT-small"

class CodeTester:
    def __init__(self):
        self.model_dir = Path(MODEL_DIR)
        self.model = None
        self.tokenizer = None
        
    def load(self):
        if not self.model_dir.exists():
            print(f"❌ ERROR: {MODEL_DIR} not found!")
            print("Run your training script first!")
            sys.exit(1)
            
        print("🔄 Loading model...")
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        self.model = PeftModel.from_pretrained(base, self.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("✅ Model loaded!")
        
    def generate(self, prompt):
        prompt = f"Write Python function: {prompt}\n```
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=80, temperature=0.1,
                do_sample=True, pad_token_id=self.tokenizer.eos_token_id
            )
            
        text = self.tokenizer.decode(outputs, skip_special_tokens=True)
        
        # Clean extraction
        if "def " in text:
            code = text.split("def ")[-1].split("```")[0].strip()
        else:
            code = text.split("\n")[-3:]  # Last few lines
        
        # Fix garbage
        code = re.sub(r'from reverse|import reverse', '', code)
        code = code.replace('reverse.reverse', 's[::-1]')
        
        return f"``````"
        
    def test(self, code):
        try:
            exec(code)
            return "✅ OK"
        except Exception as e:
            return f"❌ {str(e)[:40]}"

def main():
    tester = CodeTester()
    tester.load()
    
    tests = [
        "reverse a string",
        "sort list alphabetically", 
        "calculate fibonacci(n)",
        "read csv first column unique values"
    ]
    
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    
    for prompt in tests:
        print(f"\n📝 {prompt}")
        code_block = tester.generate(prompt)
        print(code_block)
        
        code = re.search(r'``````', code_block, re.DOTALL)
        if code:
            print(tester.test(code.group(1)))
        print("-"*30)
    
    print("\n🎯 Live test (quit to exit):")
    while True:
        prompt = input("> ")
        if prompt.lower() in ['quit', 'q']:
            break
        print(tester.generate(prompt))

if __name__ == "__main__":
    main()
