from fastapi import FastAPI, Body
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import requests
import os

MODEL_PATH = "model"  # local directory inside container

app = FastAPI()

# Load local model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)


@app.post("/generate")
async def generate(prompt: str = Body(..., embed=True)):
    # Step 1: Generate response using local LLM
    local_result = generator(prompt, max_length=200, num_return_sequences=1)
    local_text = local_result[0]["generated_text"]

    # Step 2: Get Gemini API key from env
    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        return {
            "error": "Gemini API key not set. Please provide GEMINI_API_KEY environment variable."
        }

    # Step 3: Send local model output to Gemini API
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-1.5-flash-latest:generateContent?key={API_KEY}"
    )
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": local_text}]}]}

    try:
        resp = requests.post(url, headers=headers, json=data, timeout=30)
        gemini_output = resp.json()
    except Exception as e:
        return {"error": f"Failed to call Gemini API: {str(e)}"}

    # Step 4: Extract Gemini reply safely
    if "candidates" in gemini_output:
        gemini_text = gemini_output["candidates"][0]["content"]["parts"][0]["text"]
    else:
        gemini_text = f"Gemini API error: {gemini_output}"

    # Step 5: Return structured JSON response
    return {
        "question": prompt,
        "local_llm_output": local_text,
        "gemini_output": gemini_text,
    }
