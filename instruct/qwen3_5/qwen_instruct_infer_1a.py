import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_from_disk
from tqdm import tqdm

base_model_id = "Qwen/Qwen3.5-9B"
adapter_path = "./qwen3_5-it-glossary-1a-best-model" 

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

tokenizer.chat_template = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n'}}"
    "{% if message['role'] == 'assistant' %}"
    "{% generation %}{{ message['content'] }}{{'<|im_end|>\n'}}{% endgeneration %}"
    "{% else %}"
    "{{ message['content'] }}{{'<|im_end|>\n'}}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)
tokenizer.padding_side = "right"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="cuda:0",
    # device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa"
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

print("Loading test dataset...")
test_dataset = load_from_disk("../../jumbled_data/messages_jumbled_data_instruct_test_1a")
# test_dataset=test_dataset.select(range(2))

output_file = "qwen_jumbled_results_1a.jsonl"
open(output_file, "w").close()

results = []


# print(test_dataset[0]["messages"][:-1])

print("Running inference...")
for item in tqdm(test_dataset):

    inputs = tokenizer.apply_chat_template(
        item["messages"][:-1],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024, 
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    
    input_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_length:]
    response_str = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    expected_str = item["messages"][-1]["content"]
    try:
        expected_dict = json.loads(expected_str)["glossary"]
    except json.JSONDecodeError:
        expected_dict = expected_str
        
    clean_response_str = response_str.strip()
    if clean_response_str.startswith("```json"):
        clean_response_str = clean_response_str[7:]
    if clean_response_str.endswith("```"):
        clean_response_str = clean_response_str[:-3]
    clean_response_str = clean_response_str.strip()

    try:
        generated_dict = json.loads(clean_response_str)["glossary"]
    except json.JSONDecodeError:
        generated_dict = clean_response_str 
    
    step_result = {
        "prompt": item["messages"][0]["content"],
        "expected_glossary": expected_dict,
        "generated_glossary": generated_dict,
        "raw_generated_text": response_str 
    }

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(step_result, ensure_ascii=False) + "\n")


print("Evaluation complete. Check final_test_results_detailed_1a.json")