import os, sys, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

with open(".env") as f:
    for line in f:
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()

from bench.datasets.godclass import GodClassDataset
from llm_gateway.models import PromptRenderer

ds = GodClassDataset()
ds.load_problems()
prompt_text = ds.format_prompt("GC_001")

import google.generativeai as genai
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")
cfg = genai.types.GenerationConfig(temperature=0.0, max_output_tokens=4096)
r = model.generate_content(prompt_text, generation_config=cfg)

raw = r.text or ""
print(f"raw_text length: {len(raw)}")
print(f"finish_reason: {r.candidates[0].finish_reason}")
print(f"tokens: {r.usage_metadata}")

extracted = PromptRenderer.extract_code_blocks(raw)
print(f"\nextracted_code length: {len(extracted)}")
print(f"\nextracted first 300 chars:\n{extracted[:300]}")
print(f"\nextracted last 100 chars:\n{extracted[-100:]}")
