import pandas as pd

dataset = pd.read_csv("datasets/dataset.csv", sep=",")

train = dataset[(dataset["source"] == "train")
    | (dataset["source"] == "augmentation")
    | (dataset["source"] == "generation")]
test = dataset[dataset["source"] == "test"]
valid = dataset[dataset["source"] == "valid"]

train.drop(columns=["source"]).to_csv("datasets/dataset_train.csv", sep=",", index=False)
test.drop(columns=["source"]).to_csv("datasets/dataset_test.csv", sep=",", index=False)
valid.drop(columns=["source"]).to_csv("datasets/dataset_valid.csv", sep=",", index=False)
