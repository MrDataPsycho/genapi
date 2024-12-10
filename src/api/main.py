# main.py

from fastapi import FastAPI, Query
from genapi.models.text import generate_text, load_text_model
import uvicorn

app = FastAPI()

@app.get("/")
def root_controller():
  return {"status": "healthy"}


@app.get("/generate/text")
def serve_language_model_controller(
  prompt: str = Query(..., description="user prompt"),
) -> str:
  pipe = load_text_model()
  output = generate_text(pipe, prompt)
  return output

if __name__ == "__main__":
  uvicorn.run("main:app", port=8000, reload=True)