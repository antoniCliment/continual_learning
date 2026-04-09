from unsloth import FastLanguageModel
import torch

model_id = "nvidia/Nemotron-Mini-4B-Instruct" #"./models/nemotron-mini-4b-rhinolume_v21/checkpoint-1000"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_id,
    max_seq_length = 4096,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)


messages = [{"role": "user", "content": "Tell me the characteristics that rhinolumes have"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# User parameters
model_inputs = tokenizer([text], return_tensors="pt", add_special_tokens=False).to(model.device)
eos = tokenizer.eos_token_id
terminators = [eos]
if "<extra_id_1>" in tokenizer.get_vocab():
    terminators.append(tokenizer.convert_tokens_to_ids("<extra_id_1>"))

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1024,
    do_sample=True, 
    temperature=0.2, 
    top_p=0.9,
    repetition_penalty=1.05, 
    no_repeat_ngram_size=4,
    eos_token_id=terminators,
    pad_token_id=tokenizer.pad_token_id or eos,
)

output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
print(f"Output: {repr(generated_text)}")