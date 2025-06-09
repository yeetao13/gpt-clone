import tiktoken
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.SpamDataset import SpamDataset

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

print(train_dataset.max_length)
print(val_dataset.max_length)
print(test_dataset.max_length)