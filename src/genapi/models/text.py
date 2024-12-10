import torch
from transformers import Pipeline, pipeline

system_prompt = """Your name is FastAPI bot.
You are very helpful chatbot responsible for teaching FastAPI to your users.
Always respond in markdown.
"""

def load_text_model() -> Pipeline:
    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.bfloat16
    )
    return pipe

def generate_text(pipe: Pipeline, prompt: str, temp: float = 0.7) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    prompt = pipe.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prediction = pipe(
        prompt,
        temperature=temp,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )
    output = prediction[0]["generated_text"].split("</s>\n<|assistant|>\n")[-1]
    return output


if __name__ == "__main__":
    prompt = "How to set up a FastAPI project?"
    pipe = load_text_model()
    print(generate_text(pipe, prompt))