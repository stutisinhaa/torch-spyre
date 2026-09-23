import torch
from transformers import AutoTokenizer

from hf_adapters import AutoSpyreModelForCausalLM, encode_prompts

MODEL_ID = "google/gemma-4-26B-A4B-it"

model = AutoSpyreModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

inputs = encode_prompts(tokenizer, ["What is the capital city of France?"], padding_side="left")

with torch.no_grad():
    sequences = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False,
    )

prompt_len = inputs["input_ids"].shape[1]
output_text = tokenizer.decode(sequences[0, prompt_len:], skip_special_tokens=True)
print(output_text)
