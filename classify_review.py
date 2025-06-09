import torch
import tiktoken
from utils.ClassifyReview import classify_review
from model.GPTModel import GPTModel
from data.SpamDataset import SpamDataset

BASE_CONFIG = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "drop_rate_attn": 0.1,
    "drop_rate_shortcut": 0.1,
    "drop_rate_emb": 0.1,       # Dropout rate
    "qkv_bias": True        # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-small (124M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

gpt = GPTModel(BASE_CONFIG)

num_classes = 2
gpt.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"],
    out_features=num_classes
)

tokenizer = tiktoken.get_encoding("gpt2")

train_dataset = SpamDataset(
    csv_file="D:\\Learning\\deep-learning-learn\\gpt-clone\\data\\dataset\\train.csv",
    max_length=None,
    tokenizer=tokenizer
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt.to(device)

model_state_dict = torch.load("review_classifier.pth", map_location=device)
gpt.load_state_dict(model_state_dict)

text_1 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)
print(classify_review(
    text_1, gpt, tokenizer, device, max_length=train_dataset.max_length
))