import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download
# it downloads the latest revision by default
snapshot_download(repo_id="openai-community/gpt2-large", local_dir='/pubshare/fwk/huggingface/models/openai-community/gpt2-large', repo_type='model')
# snapshot_download(repo_id="openai-community/gpt2", local_dir='/pubshare/fwk/huggingface/models/openai-community/gpt2', repo_type='model')
# snapshot_download(repo_id="siebert/sentiment-roberta-large-english", local_dir='/pubshare/fwk/huggingface/models/siebert/sentiment-roberta-large-english', repo_type='model')
# snapshot_download(repo_id="google-bert/bert-base-cased", local_dir='/pubshare/fwk/huggingface/models/google-bert/bert-base-cased', repo_type='model')
# snapshot_download(repo_id="yelp_review_full", local_dir='/pubshare/fwk/huggingface/datasets', repo_type='dataset')
# snapshot_download(repo_id="imdb", local_dir='/pubshare/fwk/huggingface/datasets/imdb', repo_type='dataset')
