import torch
import sys
import os
import tiktoken
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.GPTModel import GPTModel
from config.gpt_config import GPT_CONFIG_124M

torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)

model = GPTModel(GPT_CONFIG_124M)
out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)