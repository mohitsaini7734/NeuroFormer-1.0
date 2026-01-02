import random

INPUT_FILE = "./data/dialogues.txt"
TRAIN_FILE = "./data/train.txt"
VAL_FILE = "./data/val.txt"
VAL_RATIO = 0.1

# 1. Read full file
with open(INPUT_FILE, encoding="utf-8") as f:
    text = f.read().strip()

# 2. Split by conversation
conversations = text.split("\n\n")
random.shuffle(conversations)

# 3. Split
val_size = int(len(conversations) * VAL_RATIO)
val_convs = conversations[:val_size]
train_convs = conversations[val_size:]

# 4. Write files
with open(TRAIN_FILE, "w", encoding="utf-8") as f:
    f.write("\n\n".join(train_convs))

with open(VAL_FILE, "w", encoding="utf-8") as f:
    f.write("\n\n".join(val_convs))

print(f"Train conversations: {len(train_convs)}")
print(f"Val conversations: {len(val_convs)}")