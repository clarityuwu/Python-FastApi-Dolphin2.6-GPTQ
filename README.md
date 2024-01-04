# Tutorial how to use this python script in a jupyter notebook

This tutorial will provide you a server endpoint for generating text generation over the web with you're models.


## How to install it in a linux server with a jupyter notebook instance

First go to you're servers kernerl and download you're model with :

```
git lfs install
git clone https://huggingface.co/TheBloke/Mixtral-8x7B-v0.1-GPTQ (for exemple but replace it with any other gptq models)
```

Make sure to install the requirements in you're jupyter notebook and execute the script with you're custom directory and port (depends of what http port is open on you're server)

And BOOM ! You have a fully custom server with you're gtpq model for test it just make a request like that :

```
http://79.123.123.93:4000/generate (ip exemple)

body exemple :

{
    "prompt": "<|im_start|>user hello<|im_end|> <|im_start|>assistant",
    "max_new_tokens": 256
}

or with custom parameters :

{
    "temperature': 0.7,
    "top_k": 50,
    "top_p": 0.9,
    "top_a": 0.0,
    "token_repetition_penalty': 1.05,
    "prompt": '<|im_start|>user Hello ai !<|im_end|> <|im_start|>assistant',
    "max_new_tokens": 256
}
```
