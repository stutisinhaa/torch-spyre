
from hf_adapters import AutoSpyreModelForCausalLM
from transformers import AutoTokenizer

MODEL_ID = "google/gemma-4-26B-A4B-it"

model = AutoSpyreModelForCausalLM.from_pretrained(MODEL_ID)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
inputs = tok(["hello"], return_tensors="pt", padding=True)
model.generate(**inputs, max_new_tokens=2)
