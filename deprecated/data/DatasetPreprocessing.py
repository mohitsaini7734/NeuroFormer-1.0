# Importing Libraries
# from datasets import load_dataset
# from dotenv import load_dotenv
import os
import re

# load_dotenv()

token = os.getenv("HF_TOKEN")

def clean_text(text):
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text).strip()
    text = text.encode("ascii", "ignore").decode()
    return re.sub(r"[^a-zA-Z0-9\s.,!?;:'\"()\-\n]", "", text)

with open("data/shakespeare.txt", "r", encoding="utf-8") as f:
    text = clean_text(f.read())

n = len(text)
split_idx = int(n * 0.9)

train_text = text[:split_idx]
val_text = text[split_idx:]

print(f"Train chars: {len(train_text)}, Val chars: {len(val_text)}")

with open("data/shakespeare_train.txt", "w", encoding="utf-8") as f:
    f.write(train_text)

with open("data/shakespeare_val.txt", "w", encoding="utf-8") as f:
    f.write(val_text)
    
    
# ChatData = load_dataset("li2017dailydialog/daily_dialog", trust_remote_code=True, token=token)

def format_chat_data(type, value):
    with open(f"data/chatbot_{type}.txt", "w", encoding="utf-8") as f:
        for dialog in value:
            for i, utterance in enumerate(dialog):
                cleaned = clean_text(utterance.strip())
                if cleaned:
                    tag = "<user>" if i % 2 == 0 else "<assistant>"
                    f.write(f"{tag} {cleaned}\n")
            f.write("\n")  

# Splitting Dataset into Train, Validation Set
# ChatDataTrain = ChatData["train"]["dialog"] + ChatData["test"]["dialog"] # 90% Train Data
# ChatDataVal = ChatData["validation"]["dialog"] # 10% Validation Data

# format_chat_data('train', ChatDataTrain)
# format_chat_data('valid', ChatDataVal)