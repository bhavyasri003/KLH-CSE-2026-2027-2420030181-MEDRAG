import os
import numpy as np
import pandas as pd
import torch
import faiss
from transformers import AutoTokenizer, AutoModel

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "medquad_clean.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "embeddings"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "medquad_embeddings.npy"
)

BATCH_SIZE = 16
MAX_LENGTH = 512


# --------------------------------------------------
# Mean Pooling with Attention Mask
# --------------------------------------------------

def mean_pooling(model_output, attention_mask):

    token_embeddings = model_output.last_hidden_state

    mask = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()
    ).float()

    masked_embeddings = token_embeddings * mask

    summed = torch.sum(
        masked_embeddings,
        dim=1
    )

    counts = torch.clamp(
        mask.sum(dim=1),
        min=1e-9
    )

    return summed / counts


# --------------------------------------------------
# Load BioBERT
# --------------------------------------------------

print("Loading BioBERT model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModel.from_pretrained(
    MODEL_NAME
)

model.eval()

print("BioBERT loaded successfully.")


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("\nLoading MedQuAD...")

df = pd.read_csv(DATA_FILE)

print("Records:", len(df))


# --------------------------------------------------
# Prepare retrieval text
# --------------------------------------------------

# We give BioBERT the question and answer together.
texts = (
    "Question: "
    + df["question"].fillna("").astype(str)
    + " Answer: "
    + df["answer"].fillna("").astype(str)
)


# --------------------------------------------------
# Generate embeddings in batches
# --------------------------------------------------

all_embeddings = []

print("\nGenerating BioBERT embeddings...")

for start in range(0, len(texts), BATCH_SIZE):

    end = min(
        start + BATCH_SIZE,
        len(texts)
    )

    batch_texts = texts.iloc[start:end].tolist()

    encoded = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(**encoded)

    embeddings = mean_pooling(
        outputs,
        encoded["attention_mask"]
    )

    # Normalize each vector
    embeddings = torch.nn.functional.normalize(
        embeddings,
        p=2,
        dim=1
    )

    all_embeddings.append(
        embeddings.cpu().numpy()
    )

    print(
        f"Processed {end}/{len(texts)} records"
    )


# --------------------------------------------------
# Combine embeddings
# --------------------------------------------------

embeddings = np.vstack(
    all_embeddings
).astype("float32")


# --------------------------------------------------
# Save embeddings
# --------------------------------------------------

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

np.save(
    OUTPUT_FILE,
    embeddings
)


# --------------------------------------------------
# Final information
# --------------------------------------------------

print("\n======================================")
print("BIOBERT EMBEDDINGS COMPLETED")
print("======================================")

print(
    "Number of records :",
    embeddings.shape[0]
)

print(
    "Embedding shape   :",
    embeddings.shape
)

print("\nSaved to:")
print(OUTPUT_FILE)