import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import json

# Define paths
app_directory = os.path.dirname(os.path.realpath(__file__))
dataset_path = os.path.join(app_directory, "data", "dataset.json")  # Update file extension to .json

# Load GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Define batch size and accumulate steps
batch_size = 4
accumulate_steps = 8  # Accumulate gradients over this many steps before performing optimization

# Define the collate_fn function
def collate_fn(batch):
    if not batch or not any(batch):
        return {"input_ids": torch.tensor([], dtype=torch.long),
                "attention_mask": torch.tensor([], dtype=torch.long),
                "labels": torch.tensor([], dtype=torch.long)}

    max_len = max([max(len(x[0]), len(x[1])) for x in batch])
    input_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.zeros((len(batch), max_len), dtype=torch.long)

    for i, (user_tokens, response_tokens) in enumerate(batch):
        input_ids[i, :len(user_tokens)] = torch.tensor(user_tokens, dtype=torch.long)
        attention_mask[i, :len(user_tokens)] = 1
        labels[i, :len(response_tokens)] = torch.tensor(response_tokens, dtype=torch.long)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# Define a custom dataset class
class ConversationDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.conversations = []

        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        for entry in data:
            user_input = entry['user']
            ai_response = entry['assistant']
            user_input_tokens = tokenizer.encode(user_input, truncation=True, max_length=self.max_length)
            ai_response_tokens = tokenizer.encode(ai_response, truncation=True, max_length=self.max_length)
            self.conversations.append((user_input_tokens, ai_response_tokens))

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        return self.conversations[idx]

# Initialize the dataset
conversation_dataset = ConversationDataset(tokenizer, dataset_path)

# Check if the dataset is empty
if len(conversation_dataset) == 0:
    print("Dataset is empty.")
else:
    # Continue with DataLoader initialization, using dynamic batching
    dataloader = DataLoader(conversation_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

learning_rate = 5e-5  # You can adjust this value as needed

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)  # Adjust step_size and gamma as needed

# Train the model
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        attention_mask = batch["attention_mask"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        labels = batch["labels"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits.view(-1, model.config.vocab_size), labels.view(-1))

        loss.backward()

        if (step + 1) % accumulate_steps == 0 or step == len(dataloader) - 1:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

    scheduler.step()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

# Save the model with configuration
model_save_path = os.path.join(app_directory, "gpt2_conversational_model.pth")
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
}, model_save_path)
