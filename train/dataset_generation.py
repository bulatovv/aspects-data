import pandas as pd

AUGMENTED_PATH = "augmented_2.csv"
GENERATED_PATH = "generated_1.csv"
TEST_PATH = "test.csv"
TRAIN_PATH = "train.csv"
VALID_PATH = "valid.csv"
OUT_PATH = "dataset.csv"

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

dest_nan = (len(melted[(melted["source"] == "train") & (melted["sentiment"] == 1)])
    + len(melted[(melted["source"] == "test") & (melted["sentiment"] == 2)])
    + len(melted[(melted["source"] == "test") & (melted["sentiment"] == 3)])) / 3
dest_nan = (len(melted[(melted["source"] == "valid") & (melted["sentiment"] == 0)])
    + len(melted[(melted["source"] == "test") & (melted["sentiment"] == 0)])) / 2
train_nan = len(melted[(melted["source"] == "train") & (melted["sentiment"] == 0)])
unwanted_nan = train_nan - dest_nan

if unwanted_nan > 0:
    melted.loc[melted["source"] == "train", "source"] = "_train"
    melted = melted.sample(frac=1).sort_values(by=["source", "sentiment"])
    melted = melted[int(unwanted_nan):]
    melted.loc[melted["source"] == "_train", "source"] = "train"

melted.to_csv(OUT_PATH, sep=",", index=False)
