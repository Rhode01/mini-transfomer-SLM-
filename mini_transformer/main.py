from src.data_loader import load_sts_data
from src.tokenizer import SimpleTokenizer

sentences1, sentences2, labels = load_sts_data("train")

tokenizer = SimpleTokenizer()
tokenizer.fit([sentences1, sentences2])
