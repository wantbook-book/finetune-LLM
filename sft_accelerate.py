from ft_datasets.imdb import IMDBDataset
from transformers import AdamW, pipeline
from datasets import load_dataset

from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer,AutoModelForSequenceClassification, BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader
from utils.io_utils import save
import torch
from pathlib import Path

import os

import logging
from accelerate import Accelerator, DistributedDataParallelKwargs
# 设置环境变量以限制使用的GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# 设置日志记录
def setup_logging(log_file):
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


# 准备情感分类函数
# def classify_sentiment(texts, model, tokenizer, device):
#     model.eval()
#     inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     logits = outputs.logits
#     predictions = torch.argmax(logits, dim=-1)
#     return predictions

def classify_sentiment(texts, model, tokenizer, device):
    model.eval()
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    predictions = torch.argmax(logits, dim=-1)
    
    labels = ["NEGATIVE", "POSITIVE"]
    results = []
    for i in range(len(predictions)):
        label = labels[predictions[i].item()]
        probability = probabilities[i][predictions[i]].item()
        results.append({
            "label": label,
            "score": probability
        })
        
    return results

def evaluate(model, tokenizer, dataloader, sentiment_model, sentiment_tokenizer, device)->tuple[list[float], list[float]]:
    # device = torch.device('cuda:2')
    model.eval()
    total_score = 0
    count = 0
    positive_score = []
    negative_score = [] 
    for batch in dataloader:
        print(count)
        count+=1
        inputs = batch["input_ids"][:,:6]
        attention_mask = batch["attention_mask"][:, :6]
        with torch.no_grad():
            outputs = model.module.generate(inputs, attention_mask=attention_mask, max_length=510, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        for output in outputs:
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            # sentiment = sentiment_classifier(generated_text)[0]
            sentiment = classify_sentiment([generated_text], sentiment_model, sentiment_tokenizer, device)[0]
            if sentiment['label'] == 'POSITIVE':
                # total_score += sentiment['score']
                positive_score.append(sentiment['score'])
            else:
                negative_score.append(sentiment['score'])
            # count += 1
    # avg_score = total_score / count if count > 0 else 0
    # avg_pos_score = sum(positive_score) / len(positive_score) if len(positive_score) > 0 else 0
    # avg_neg_score = sum(negative_score) / len(negative_score) if len(negative_score) > 0 else 0
    # return avg_pos_score, avg_neg_score
    return positive_score, negative_score
def train(model, tokenizer, optimizer, train_dataloader, test_dataloader, sentiment_model, sentiment_tokenizer, device, epochs=1, lr=5e-5, eval_steps=20, first_eval=True):
    model.train()
    # optimizer = AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for step, batch in enumerate(train_dataloader):
            inputs = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            outputs = model(inputs, attention_mask=attention_mask, labels=inputs)
            loss = outputs.loss.mean()
            optimizer.zero_grad()
            # loss.backward()
            accelerator.backward(loss)
            
            optimizer.step()
            if step % eval_steps == 0 and (step > 0 or first_eval):
                positive_score, negative_score = evaluate(model, tokenizer, test_dataloader, sentiment_model, sentiment_tokenizer, device)
                avg_positve_score = sum(positive_score)/len(positive_score) if len(positive_score)>0 else 0
                avg_negative_score = sum(negative_score)/len(negative_score) if len(negative_score)>0 else 0
                eval_logstr = f"Epoch {epoch + 1}, Step {step+1}, Positive Avg Score: {avg_positve_score:.2f}, Positive Num: {len(positive_score)}, Negative Score: {avg_negative_score:.2f}, Neg Num: {len(negative_score)}"
                logging.info(eval_logstr)
                # print(eval_logstr)
                # print(f"Epoch {epoch + 1}, Step {step+1}, Positive Avg Score: {avg_positve_score:.2f}, Positive Num: {len(positive_score)}, Negative Score: {avg_negative_score:.2f}, Neg Num: {len(negative_score)}")
            train_logstr = f"Epoch {epoch + 1}, Step {step+1}, Train Loss: {loss.item()}"
            logging.info(train_logstr)
            # print(train_logstr)
            # print(f"Epoch {epoch + 1}, Step {step+1}, Train Loss: {loss.item()}")

# 加载模型
model_path = '/pubshare/fwk/huggingface/models/openai-community/gpt2-large'
# 模型加载
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(model_path)

# 加载 sentiment classifier
sentiment_model_path = '/pubshare/fwk/huggingface/models/siebert/sentiment-roberta-large-english'
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path)
if sentiment_tokenizer.pad_token_id is None:
    sentiment_tokenizer.pad_token_id = sentiment_tokenizer.eos_token_id
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path)

# 加载数据集
data_path = '/pubshare/fwk/huggingface/datasets/imdb/plain_text'
# 加载 IMDb 数据集
dataset = load_dataset(data_path)
# 过滤正面评论
positive_reviews = dataset['train'].filter(lambda x: x['label'] == 1)
# 分割为训练集和验证集
train_test_split = positive_reviews.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']
print('train dataset size:', len(train_dataset))
print('test dataset size:', len(test_dataset))
# 数据集准备
train_dataset = IMDBDataset(train_dataset, tokenizer)
test_dataset = IMDBDataset(test_dataset, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 日志
log_file = 'imdb_sft_gpt2large4.txt'
setup_logging(log_file)

# accelerator 实例化
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
device = accelerator.device

# 优化器
optimizer = AdamW(model.parameters(), lr=5e-5)
model, optimizer, train_loader, test_loader, sentiment_model = accelerator.prepare(
    model, optimizer, train_loader, test_loader, sentiment_model
)

# 开始训练
train(
    model=model,
    tokenizer=tokenizer,
    optimizer=optimizer,
    train_dataloader=train_loader,
    test_dataloader=test_loader,
    sentiment_model=sentiment_model,
    sentiment_tokenizer=sentiment_tokenizer,
    eval_steps=50,
    epochs=3,
    device=device
)

# 保存模型
if accelerator.is_main_process:
    save(model.module, tokenizer, output_dir='/pubshare/fwk/training_results/imdb_sft_gpt2_large')
