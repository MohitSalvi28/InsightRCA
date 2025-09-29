# app.py
from fastapi import FastAPI, Body
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

MODEL_PATH = "model"  # local directory inside container

app = FastAPI()

# Load model + tokenizer from local dir
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

@app.post("/generate")
async def generate(prompt: str = Body(..., embed=True)):
    result = generator(prompt, max_length=200, num_return_sequences=1)
    return {"prompt": prompt, "output": result[0]["generated_text"]}
