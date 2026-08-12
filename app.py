"""
Single-file Text Summarizer App (FastAPI + T5 + inline HTML front end)

Run with:
    uvicorn app_single_file:app --reload

Requires a fine-tuned model + tokenizer saved at ./saved_summary_model
(produced by running text_summarizer.ipynb).
"""

import re

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

# device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)


class DialogueInput(BaseModel):
    dialogue: str


def clean_data(text):
    text = re.sub(r"\r\n", " ", text)  # lines
    text = re.sub(r"\s+", " ", text)  # spaces
    text = re.sub(r"<.*?>", " ", text)  # html tags <p> <h1>
    text = text.strip().lower()
    return text


def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_data(dialogue)  # clean

    # tokenize
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # generate the summary => token ids
    model.to(device)
    targets = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )

    # decode our output
    summary = tokenizer.decode(targets[0], skip_special_tokens=True)  # EOS, SEP
    return summary


# API endpoints
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}


HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Text Summarizer</title>
<style>
  body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
    background-color: #FFF7ED;
    color: #333;
  }
  .container {
    max-width: 600px;
    margin: 50px auto;
    padding: 20px;
    background: #fff;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  }
  h1 {
    text-align: center;
    color: #FB923C;
  }
  h3 {
    text-align: center;
    margin-top: -20px;
    margin-bottom: 40px;
  }
  form {
    display: flex;
    flex-direction: column;
    gap: 15px;
  }
  textarea {
    height: 150px;
    padding: 10px;
    border: 1px solid #ccc;
    border-radius: 5px;
    font-size: 16px;
  }
  button {
    background-color: #FB923C;
    color: white;
    border: none;
    padding: 10px 15px;
    font-size: 16px;
    cursor: pointer;
    border-radius: 5px;
  }
  button:hover {
    background-color: #F97316;
  }
  #summary-output {
    margin-top: 20px;
    background: #FFEDD5;
    padding: 15px;
    border-radius: 5px;
    border: 1px solid #ccc;
  }
  #summary-heading {
    margin-top: 0px;
    margin-bottom: 0px;
  }
</style>
</head>
<body>
<div class="container">
  <h1>Text Summarizer</h1>
  <h3>using HuggingFace Transformer</h3>
  <p>Write or Paste your content below for quick summarization.</p>
  <form id="summarization-form">
    <textarea id="dialogue-input" placeholder="Enter your content here..." required></textarea>
    <button type="submit">Summarize</button>
  </form>
  <div id="summary-output">
    <h3 id="summary-heading">Content Summary</h3>
    <p id="summary-text"></p>
  </div>
</div>

<script>
document.getElementById("summarization-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const dialogueInput = document.getElementById("dialogue-input");
  const summaryText = document.getElementById("summary-text");
  const submitButton = e.target.querySelector("button");
  const dialogue = dialogueInput.value.trim();

  if (!dialogue) return;

  summaryText.innerText = "Processing...";
  submitButton.disabled = true;

  try {
    const response = await fetch("/summarize/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dialogue }),
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();
    summaryText.innerText = data.summary || "No summary returned.";
  } catch (err) {
    summaryText.innerText = `Error: ${err.message}`;
  } finally {
    submitButton.disabled = false;
  }
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_PAGE
