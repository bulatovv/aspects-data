import pandas as pd

AUGMENTED_PATH = "datasets/augmented_2.csv"
GENERATED_PATH = "datasets/generated_1.csv"
TEST_PATH = "datasets/test.csv"
TRAIN_PATH = "datasets/train.csv"
VALID_PATH = "datasets/valid.csv"
OUT_PATH = "datasets/dataset.csv"

augmented = pd.read_csv(AUGMENTED_PATH, sep=",")
generated = pd.read_csv(GENERATED_PATH, sep=",")
test = pd.read_csv(TEST_PATH, sep=",")
train = pd.read_csv(TRAIN_PATH, sep=",")
valid = pd.read_csv(VALID_PATH, sep=",")

augmented.insert(1, "source", "augmentation")
generated.insert(1, "source", "generation")
train.insert(1, "source", "train")
test.insert(1, "source", "test")
valid.insert(1, "source", "valid")

merged = pd.concat([augmented, generated, train, test, valid])

melted = pd.melt(merged, id_vars=["text", "source"])
melted.rename(columns={"variable": "aspect", "value": "sentiment"}, inplace=True)
melted.loc[melted["sentiment"].isna(), "sentiment"] = 0

# Удалить нули в аугментации и генерации
melted = melted[((melted["source"] != "augmentation") & (melted["source"] != "generation")) | (melted["sentiment"] != 0.0)]

dest_nan = (
    len(melted[(melted["source"] == "train") & (melted["sentiment"] == 1)])
    + len(melted[(melted["source"] == "train") & (melted["sentiment"] == 2)])
    + len(melted[(melted["source"] == "train") & (melted["sentiment"] == 3)])
    + len(melted[melted["source"] == "augmentation"])
    + len(melted[melted["source"] == "generation"])
    ) / 3
train_nan = len(melted[(melted["source"] == "train") & (melted["sentiment"] == 0)])
unwanted_nan = train_nan - dest_nan

if unwanted_nan > 0:
    melted.loc[melted["source"] == "train", "source"] = "_train"
    melted = melted.sample(frac=1, random_state=42).sort_values(by=["source", "sentiment"])
    melted = melted[int(unwanted_nan):]
    melted.loc[melted["source"] == "_train", "source"] = "train"

melted.to_csv(OUT_PATH, sep=",", index=False)
