from pathlib import Path
from datetime import datetime

def save(model, tokenizer, output_dir=''):
    output_dir = Path(output_dir)
    if output_dir.exists():
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_dir = output_dir / now
    output_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == '__main__':
    save(None, None, output_dir='/pubshare/fwk/training_results/imdb_sft_gpt2_large')