# import torch
# import json
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from peft import PeftModel
# from datasets import load_from_disk
# from tqdm import tqdm
# import re

# base_model_id = "Qwen/Qwen3.5-9B"
# adapter_path = "./qwen3_5-it-glossary-1b-best-model" 

# print("Loading tokenizer and model...")
# tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# tokenizer.chat_template = (
#     "{% for message in messages %}"
#     "{{'<|im_start|>' + message['role'] + '\n'}}"
#     "{% if message['role'] == 'assistant' %}"
#     "{% generation %}{{ message['content'] }}{{'<|im_end|>\n'}}{% endgeneration %}"
#     "{% else %}"
#     "{{ message['content'] }}{{'<|im_end|>\n'}}"
#     "{% endif %}"
#     "{% endfor %}"
#     "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
# )
# tokenizer.padding_side = "right"

# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# base_model = AutoModelForCausalLM.from_pretrained(
#     base_model_id,
#     quantization_config=bnb_config,
#     # device_map="cuda:1",
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="sdpa"
# )

# model = PeftModel.from_pretrained(base_model, adapter_path)
# model.eval()

# print("Loading test dataset...")
# test_dataset = load_from_disk("../../jumbled_data/jumbled_pc_data_instruct_test_1b")


# output_file = "qwen_jumbled_results_1b.jsonl"
    
# results = []

# print("Running inference...")
# for idx,item in enumerate(tqdm(test_dataset)):
#     inputs = tokenizer.apply_chat_template(
#         item["messages"][:-1],
#         add_generation_prompt=True,
#         return_tensors="pt",
#         return_dict=True
#     ).to(model.device)
    
#     with torch.inference_mode():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=900, 
#             do_sample=False,
#             # do_sample=True,
#             # temperature=0.7,
#             # top_p=0.9,
#             # repetition_penalty=1.4,
#             # use_cache=True,
#             pad_token_id=tokenizer.pad_token_id
#         )
    
#     input_length = inputs["input_ids"].shape[1]
#     generated_ids = outputs[0][input_length:]
#     response_str = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
#     expected_str = item["messages"][-1]["content"]

#     expected_reasoning_str=expected_str.split("### FINAL GENERATION:")[0].strip().strip("\n")
#     pattern = r'\[[^\]]*\]'
#     expected_padacheda_list=re.findall(pattern, expected_reasoning_str)

#     expected_glossary_str=expected_str.split("### FINAL GENERATION:")[-1].strip().strip("\n")
    
#     try:
#         if expected_glossary_str.startswith("```json"):
#             expected_glossary_str = expected_glossary_str[7:]
#         if expected_glossary_str.endswith("```"):
#             expected_glossary_str = expected_glossary_str[:-3]
#         expected_dict = json.loads(expected_glossary_str)["glossary"]
        
#     except json.JSONDecodeError:
#         expected_dict = expected_glossary_str


#     generated_reasoning_str=response_str.split("### FINAL GENERATION:")[0].strip().strip("\n")
#     pattern = r'\[[^\]]*\]'
#     generated_padacheda_list=re.findall(pattern, generated_reasoning_str)

#     generated_glossary_str=response_str.split("### FINAL GENERATION:")[-1].strip().strip("\n")

#     try:
#         if generated_glossary_str.startswith("```json"):
#             generated_glossary_str = generated_glossary_str[7:]
#         if generated_glossary_str.endswith("```"):
#             generated_glossary_str = generated_glossary_str[:-3]
#         generated_dict = json.loads(generated_glossary_str)["glossary"]
#     except json.JSONDecodeError:
#         generated_dict = generated_glossary_str 
    
#     step_result = {
#         "prompt": item["messages"][0]["content"],
#         "expected_padaccheda" : expected_padacheda_list,
#         "expected_glossary": expected_dict,
#         "generated_padaccheda":generated_padacheda_list,
#         "generated_glossary": generated_dict,
#         "raw_generated_text": response_str 
#     }


# print("Evaluation complete. Check final_test_results_detailed_1b.json")











