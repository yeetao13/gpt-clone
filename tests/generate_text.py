import torch
import tiktoken
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.GPTModel import GPTModel
from config.gpt_config import GPT_CONFIG_124M

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")

model = GPTModel(GPT_CONFIG_124M)

start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded) # this is a list
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape (after unsqeeze):", encoded_tensor.shape)

model.eval()
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output:", out)
print("Output length:", len(out[0]))

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)