import tiktoken
import torch
from torch.utils.data import DataLoader

from data.SpamDataset import SpamDataset
from model.GPTModel import GPTModel
from utils.TextTokenIDConversion import text_to_token_ids, token_ids_to_text, generate_text_simple
from utils.GenerateText import generate
from utils.CrossEntropyClassification import calc_accuracy_loader, calc_loss_loader
from trainer.trainer_classification import train_classifier_simple

tokenizer = tiktoken.get_encoding("gpt2")

train_dataset = SpamDataset(
    csv_file="D:\\Learning\\deep-learning-learn\\gpt-clone\\data\\dataset\\train.csv",
    max_length=None,
    tokenizer=tokenizer
)

val_dataset = SpamDataset(
    csv_file="D:\\Learning\\deep-learning-learn\\gpt-clone\\data\\dataset\\validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)
test_dataset = SpamDataset(
    csv_file="D:\\Learning\\deep-learning-learn\\gpt-clone\\data\\dataset\\test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

# for input_batch, target_batch in train_loader:
#     pass
# print("Input batch dimensions:", input_batch.shape)
# print("Label batch dimensions", target_batch.shape)
# print(f"{len(train_loader)} training batches")
# print(f"{len(val_loader)} validation batches")
# print(f"{len(test_loader)} test batches")

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

file_name = "gpt2-small-124M.pth"

gpt = GPTModel(BASE_CONFIG)
gpt.load_state_dict(torch.load(file_name, weights_only=True))
gpt.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt.to(device)

torch.manual_seed(123)

tokenizer = tiktoken.get_encoding("gpt2")

# text_2 = (
# "Is the following text 'spam'? Answer with 'yes' or 'no':"
# " 'You are a winner you have been specially"
# " selected to receive $1000 cash or a $2000 award.'"
# )
# token_ids = generate_text_simple(
#     model=gpt,
#     idx=text_to_token_ids(text_2, tokenizer),
#     max_new_tokens=23,
#     context_size=BASE_CONFIG["context_length"]
# )
# print(token_ids_to_text(token_ids, tokenizer))

for param in gpt.parameters():
    param.requires_grad = False

num_classes = 2
gpt.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"],
    out_features=num_classes
)

for param in gpt.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in gpt.final_norm.parameters():
    param.requires_grad = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt.to(device)
torch.manual_seed(123)
train_accuracy = calc_accuracy_loader(
    train_loader, gpt, device, num_batches=10
)
val_accuracy = calc_accuracy_loader(
    val_loader, gpt, device, num_batches=10
)
test_accuracy = calc_accuracy_loader(
    test_loader, gpt, device, num_batches=10
)
print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")

with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, gpt, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, gpt, device, num_batches=5)
    test_loss = calc_loss_loader(test_loader, gpt, device, num_batches=5)
print(f"Training loss: {train_loss:.3f}")
print(f"Validation loss: {val_loss:.3f}")
print(f"Test loss: {test_loss:.3f}")


optimizer = torch.optim.AdamW(gpt.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 5
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
    gpt, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=50,
    eval_iter=5
)

print(f"Training completed.")