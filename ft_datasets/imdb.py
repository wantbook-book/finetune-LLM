import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
class IMDBDataset(Dataset):
    def __init__(self, reviews, tokenizer, max_length=512):
        self.reviews = reviews
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]['text']
        inputs = self.tokenizer(review, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        return inputs


# if __name__ == '__main__':
#     tokenizer = 
#     train_dataset = IMDBDataset('/pubshare/fwk/huggingface/datasets/imdb/plain_text', tokenizer)
#     train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
