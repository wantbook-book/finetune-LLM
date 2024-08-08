import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, AdamW
from datasets import load_dataset
from accelerate import Accelerator

# 加载 IMDb 数据集
dataset = load_dataset("imdb")
positive_reviews = dataset['train'].filter(lambda x: x['label'] == 1)

# 分割为训练集和验证集
train_test_split = positive_reviews.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# 定义数据集类
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

# 初始化Accelerator
accelerator = Accelerator()

# 加载模型和分词器
model_name = '/pubshare/fwk/huggingface/models/openai-community/gpt2-large'
sentiment_model_name = "/pubshare/fwk/huggingface/models/siebert/sentiment-roberta-large-english"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
if sentiment_tokenizer.pad_token_id is None:
    sentiment_tokenizer.pad_token_id = sentiment_tokenizer.eos_token_id

# 准备数据集和数据加载器
train_dataset = IMDBDataset(train_dataset, tokenizer)
test_dataset = IMDBDataset(test_dataset, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# 使用Accelerator准备模型、优化器和数据加载器
optimizer = AdamW(model.parameters(), lr=5e-5)
model, optimizer, train_loader, test_loader, sentiment_model = accelerator.prepare(model, optimizer, train_loader, test_loader, sentiment_model)

# 训练函数
def train(model, tokenizer, train_dataloader, test_dataloader, sentiment_model, sentiment_tokenizer, accelerator, epochs=1, lr=5e-5):
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            if step % 10 == 0:
                print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item()}")

# 开始训练
train(model, tokenizer, train_loader, test_loader, sentiment_model, sentiment_tokenizer, accelerator, epochs=3)

# 生成正面情感的评论
def generate_positive_review(prefix, model, tokenizer, accelerator, max_length=50):
    model.eval()
    inputs = tokenizer(prefix, return_tensors="pt").to(accelerator.device)
    outputs = model.generate(inputs.input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
    review = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return review

# 示例生成
prefix = "The movie was"
positive_review = generate_positive_review(prefix, model, tokenizer, accelerator)
print(positive_review)
