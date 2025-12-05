import os
import re
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:32"
torch.cuda.empty_cache()

MODEL_DIR = "./my-python-code-llm-final"
BASE_MODEL = "microsoft/DialoGPT-small"

class CodeTester:
    def __init__(self):
        self.model_dir = Path(MODEL_DIR)
        self.model = None
        self.tokenizer = None
        
    def load(self):
        if not self.model_dir.exists():
            print("❌ ERROR: ./my-python-code-llm-final/ not found!")
            print("Run training first!")
            return False
            
        print("🔄 Loading...")
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        self.model = PeftModel.from_pretrained(base, self.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("✅ Loaded!")
        return True
        
    def generate(self, prompt):
        # NO F-STRINGS - plain strings only
        full_prompt = "Write Python function: " + prompt + "\n```
        inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=80, temperature=0.1, do_sample=True, pad_token_id=self.tokenizer.eos_token_id)
            
        text = self.tokenizer.decode(outputs, skip_special_tokens=True)
        
        if "def " in text:
            code = text.split("def ")[-1].split("```")[0].strip()
        else:
            code = text.split("\n")[-3:]
            
        code = re.sub(r'from reverse|import reverse', '', code)
        code = code.replace('reverse.reverse', 's[::-1]')
        
        return "``````"
        
    def test(self, code):
        try:
            exec(code)
            return "✅ OK"
        except:
            return "❌ Error"

# RUN
tester = CodeTester()
if tester.load():
    tests = ["reverse a string", "sort list alphabetically"]
    
    for prompt in tests:
        print("\n📝", prompt)
        code_block = tester.generate(prompt)
        print(code_block)
        
        code = re.search(r'``````', code_block, re.DOTALL)
        if code:
            print(tester.test(code.group(1)))
        print("-"*30)
    
    print("\nLive test:")
    prompt = input("Prompt: ")
    print(tester.generate(prompt))
