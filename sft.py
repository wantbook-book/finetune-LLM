from ft_datasets.imdb import IMDBDataset
from transformers import AdamW, pipeline
from datasets import load_dataset

from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer, BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader
from utils.io_utils import save
import torch
from pathlib import Path

# 准备情感分类函数
def classify_sentiment(texts, model, tokenizer, device):
    model.eval()
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    return predictions

def train(model, tokenizer, train_dataloader, test_dataloader, sentiment_classifier, device, log_filepath:Path, epochs=1, lr=5e-5, eval_steps=20, first_eval=True):
    log_filepath.parent.mkdir(parents=True, exist_ok=True) 
    f = open(log_filepath, 'w')
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for step, batch in enumerate(train_dataloader):
            inputs = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(inputs, attention_mask=attention_mask, labels=inputs)
            loss = outputs.loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % eval_steps == 0 and (step > 0 or first_eval):
                positive_score, negative_score = evaluate(model, tokenizer, test_dataloader, sentiment_classifier, device)
                avg_positve_score = sum(positive_score)/len(positive_score) if len(positive_score)>0 else 0
                avg_negative_score = sum(negative_score)/len(negative_score) if len(negative_score)>0 else 0
                eval_logstr = f"Epoch {epoch + 1}, Step {step+1}, Positive Avg Score: {avg_positve_score:.2f}, Positive Num: {len(positive_score)}, Negative Score: {avg_negative_score:.2f}, Neg Num: {len(negative_score)}"
                f.write(eval_logstr+'\n')
                print(eval_logstr)
                # print(f"Epoch {epoch + 1}, Step {step+1}, Positive Avg Score: {avg_positve_score:.2f}, Positive Num: {len(positive_score)}, Negative Score: {avg_negative_score:.2f}, Neg Num: {len(negative_score)}")
            train_logstr = f"Epoch {epoch + 1}, Step {step+1}, Train Loss: {loss.item()}"
            f.write(train_logstr+'\n')
            print(train_logstr)
            # print(f"Epoch {epoch + 1}, Step {step+1}, Train Loss: {loss.item()}")

def evaluate(model, tokenizer, dataloader, sentiment_classifier, device)->tuple[list[float], list[float]]:
    # device = torch.device('cuda:2')
    model.eval()
    total_score = 0
    count = 0
    positive_score = []
    negative_score = [] 
    for batch in dataloader:
        print(count)
        count+=1
        inputs = batch["input_ids"][:,:6].to(device)
        attention_mask = batch["attention_mask"][:, :6].to(device)
        with torch.no_grad():
            outputs = model.generate(inputs, attention_mask=attention_mask, max_length=510, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        for output in outputs:
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            sentiment = sentiment_classifier(generated_text)[0]
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

# 加载模型
model_path = '/pubshare/fwk/huggingface/models/openai-community/gpt2-large'
# model_path = 'gpt2'
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
print('tokenizer.pad_token_id:', tokenizer.pad_token_id)
print('tokenizer.eos_token_id:', tokenizer.eos_token_id)
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

train_dataset = IMDBDataset(train_dataset, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

test_dataset = IMDBDataset(test_dataset, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cuda:4')
print('device:', device)
model.to(device)

# 加载 sentiment classifier
# sentiment_device = torch.device('cuda:4')
sentiment_model_path = '/pubshare/fwk/huggingface/models/siebert/sentiment-roberta-large-english'
# sentiment_model = BertForSequenceClassification.from_pretrained(sentiment_model_path)
# sentiment_tokenizer = BertTokenizer.from_pretrained(sentiment_model_path)
# sentiment_model.to(sentiment_device)

sentiment_classifier = pipeline("sentiment-analysis", model=sentiment_model_path)


# 开始训练
train(
    model=model,
    tokenizer=tokenizer,
    train_dataloader=train_loader,
    test_dataloader=test_loader,
    sentiment_classifier=sentiment_classifier,
    log_filepath=Path('logs/imdb_sft_gpt2_large.log'),
    eval_steps=50,
    epochs=3,
    device=device
)


save(model, tokenizer, output_dir='/pubshare/fwk/training_results/imdb_sft_gpt2_large')