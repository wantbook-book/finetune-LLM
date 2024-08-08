import deepspeed
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
master_port = 29501
import os
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = str(master_port)

class MyDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        return input_ids, attention_mask

# 准备数据
model_path = '/pubshare/fwk/huggingface/models/openai-community/gpt2-large'
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

texts = ["Example sentence 1", "Example sentence 2", "Example sentence 3"]*1000
dataset = MyDataset(texts, tokenizer, max_length=512)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 加载模型
model = GPT2LMHeadModel.from_pretrained(model_path)

# DeepSpeed 配置文件路径
ds_config = {
    "train_batch_size": 2,
    "fp16": {
        "enabled": True
    },
    # "zero_optimization": {
    #     "stage": 3,
    #     "cpu_offload": True
    # },
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 5e-5
        }
    },
    "pipeline": {
        "enabled": True,
        "stages": 2
    },
}

# 使用 DeepSpeed 初始化模型、优化器和数据加载器
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    # dataloader=dataloader,
    # training_data=dataset,
    config_params=ds_config,
    distributed_port=master_port
)

# 定义损失函数
criterion = nn.CrossEntropyLoss()
model_engine.train()
# 训练循环
for epoch in range(3):
    for batch in dataloader:
        input_ids, attention_mask = batch
        input_ids = input_ids.to(model_engine.local_rank)
        attention_mask = attention_mask.to(model_engine.local_rank)

        optimizer.zero_grad()
        outputs = model_engine(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # 计算损失
        labels = input_ids
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        # 反向传播
        model_engine.backward(loss)
        model_engine.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")
