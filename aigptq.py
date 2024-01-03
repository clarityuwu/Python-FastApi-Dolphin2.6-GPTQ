import sys, os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import time
import nest_asyncio
import re

nest_asyncio.apply()

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()

sys.path.append(os.path.dirname(current_dir))

from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)

from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2Sampler
)

import time

# Initialize model and cache

model_directory =  "dolphin-2.6-mixtral-8x7b-GPTQ"

config = ExLlamaV2Config()
config.model_dir = model_directory
config.prepare()

model = ExLlamaV2(config)
print("Loading model: " + model_directory)

cache = ExLlamaV2Cache(model, lazy = True)
model.load_autosplit(cache)

tokenizer = ExLlamaV2Tokenizer(config)

# Initialize generator

generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

def get_default_settings():
    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = 0.85
    settings.top_k = 50
    settings.top_p = 0.8
    settings.top_a = 0.0
    settings.token_repetition_penalty = 1.05
    settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])
    return settings

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

class InputData(BaseModel):
    temperature: Optional[float] = 0.85
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.8
    top_a: Optional[float] = 0.0
    token_repetition_penalty: Optional[float] = 1.05
    prompt: Optional[str] = None
    max_new_tokens: Optional[int] = 256

class OutputData(BaseModel):
    output: str
    time_taken: float

app = FastAPI()

@app.post('/generate', response_model=OutputData)
async def generate(input_data: InputData):
    # Use the generator to generate a prompt
    settings = get_default_settings()
    # Get the prompt from the request data
    user_prompt = input_data.prompt
    if not user_prompt:
        raise HTTPException(status_code=400, detail="No prompt provided")

    # Add a system prompt to the user's prompt
    system_prompt = "<|im_start|>system you are Dolphin, a helpful Ai assistant<|im_end|>"
    prompt = system_prompt + user_prompt

    max_new_tokens = input_data.max_new_tokens

    generator.warmup()
    time_begin = time.time()

    output = generator.generate_simple(prompt, settings, max_new_tokens, seed = 1234)

    output = output.replace(system_prompt, '')
    
    pattern = r'<\|im_start\|>assistant (.*?)<\|im_end\|>'
    match = re.search(pattern, output)
    if match:
        first_response = match.group(1)
    else:
        print("No match found")
        
    time_end = time.time()
    time_total = time_end - time_begin

    # Return the generated output and the time taken
    return {'output': first_response, 'time_taken': time_total}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=4000)