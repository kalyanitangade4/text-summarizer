# Dialogue Summarizer — T5 Fine-Tuning

An abstractive text summarization tool built by fine-tuning a T5-small model (Hugging Face) on the SAMSum dialogue-summarization dataset (https://huggingface.co/datasets/samsum). Includes a text-cleaning pipeline and beam-search decoding for improved summary quality, served via a single-file FastAPI app with an inline HTML/JS front end.

## Overview

- Task: Abstractive dialogue summarization (chat/conversation -> short summary)
- Base model: t5-small
- Dataset: SAMSum (4,000 sampled training dialogues, 500 validation dialogues)
- Decoding: Beam search (num_beams=4)
- Serving: Single-file FastAPI app (app_single_file.py) with the HTML/JS front end embedded inline — no templates folder or separate index.html needed

## Pipeline

1. Data loading — reads samsum-train.csv and samsum-validation.csv
2. Sampling — random subsample (4,000 train / 500 val, random_state=42) for faster iteration
3. Cleaning — normalizes whitespace, strips HTML tags, lowercases text
4. Tokenization — T5 tokenizer, dialogue truncated/padded to 512 tokens, summary to 150 tokens
5. Fine-tuning — Hugging Face Trainer, 6 epochs, batch size 8, weight decay 0.01, 500 warmup steps
6. Inference — beam search decoding (num_beams=4, max_length=150) for generating summaries
7. Serving — FastAPI REST endpoint + inline HTML page, wrapping the fine-tuned model

## Repo Structure

.
├── text_summarizer.ipynb    # Main notebook: data prep, fine-tuning, inference
├── app_single_file.py       # Single-file FastAPI app (backend + HTML front end)
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md

## Setup

git clone https://github.com/kalyanitangade4/text-summarizer.git
cd text-summarizer
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

## Data

Download the SAMSum dataset (samsum-train.csv, samsum-validation.csv) and place them in the project root, or update the paths in the notebook. Source: Hugging Face Datasets — samsum (https://huggingface.co/datasets/samsum).

## Training

Open and run text_summarizer.ipynb end-to-end. This will:
- Clean and tokenize the data
- Fine-tune t5-small for 6 epochs
- Save the fine-tuned model + tokenizer to ./saved_summary_model

## Running the App

Once a model is saved to ./saved_summary_model:

uvicorn app_single_file:app --reload

Then open http://127.0.0.1:8000 in your browser, or POST directly to /summarize/:

curl -X POST http://127.0.0.1:8000/summarize/ \
  -H "Content-Type: application/json" \
  -d '{"dialogue": "Alice: Are we still on for lunch? Bob: Yes, 1pm at the usual place."}'

## Notes

- This is a small-scale fine-tune (t5-small, subsampled data) intended as a learning/portfolio project, not a production-grade summarizer.
- Swap t5-small for t5-base/t5-large and use the full dataset for stronger results, given enough compute.
- The ./saved_summary_model folder is gitignored (model weights are too large for a plain git push). Anyone cloning this repo needs to run the notebook first to generate it before app_single_file.py will work.

## License

MIT
