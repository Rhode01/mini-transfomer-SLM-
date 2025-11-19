from datasets import load_dataset
def load_sts_data(split="train"):
    dataset = load_dataset("glue", "stsb", split=split)
    sentences1 = dataset['sentence1']
    sentences2 = dataset['sentence2']
    labels = dataset['label']
    return sentences1, sentences2, labels
