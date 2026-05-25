import datasets
from datasets import load_dataset, Dataset
from huggingface_hub import login
import json
import re
import ast
import random

login()

def filter_sans_to_english(dataset):
    filtered_dataset=dataset.filter(lambda example: example['source_language']=='sanskrit' and example['target_language']=='english')
    return filtered_dataset

def fix_malformed_glossary(broken_dict):
    full_str = ", ".join([f"{k}: {v}" for k, v in broken_dict.items()])
    
    parts = full_str.split(":")
    
    if len(parts) <= 1:
        return broken_dict
        
    fixed_dict = {}
    current_key = parts[0].strip()
    
    for i in range(1, len(parts)):
        chunk = parts[i]
        
        if i == len(parts) - 1:
            fixed_dict[current_key] = chunk.strip()
        else:

            last_comma_idx = chunk.rfind(",")
            
            if last_comma_idx == -1:
                last_space_idx = chunk.rfind(" ")
                val = chunk[:last_space_idx].strip()
                next_key = chunk[last_space_idx:].strip()
            else:
                val = chunk[:last_comma_idx].strip()
                next_key = chunk[last_comma_idx+1:].strip()
                
            fixed_dict[current_key] = val
            current_key = next_key
            
    return fixed_dict

def parse_dirty_glossary(dirty_str):
    dirty_str = str(dirty_str).strip()
    
    try:
        clean_str = re.sub(r',\s*\}', '}', dirty_str)
        parsed = ast.literal_eval(clean_str)
        if isinstance(parsed, dict):
            return {str(k).strip(): str(v).strip() for k, v in parsed.items()}
    except Exception:
        pass

    clean_str = re.sub(r'\},?\n?\}?$', '', dirty_str)
    clean_str = clean_str.strip('{ \n\r')
    
    glossary_dict = {}
    pairs = clean_str.split(',\n')
    
    for pair in pairs:
        if ':' in pair:
            key, val = pair.split(':', 1)
            key = key.strip(' "\'')
            val = val.strip(' "\'')
            glossary_dict[key] = val
            
    return fix_malformed_glossary(glossary_dict)


def format_instruct_json(dataset):
    task_1b_messages = []
    task_1b_PC = []

    for row in dataset:
        shlok = row['shlok']
        translation = row['translation']
        clean_glossary = parse_dirty_glossary(row['glossary'])
        resolved_sequence = list(clean_glossary.keys())
        if resolved_sequence is None:
            print("Oh my habibiii")

        
        sequence_string = ", ".join(resolved_sequence)
        prompt_1b = f"### INPUT:\nYou are an expert in Sanskrit. Perform Padaccheda (word resolution) on the following Sanskrit shloka, then generate the glossary mapping using the generated Padaccheda and provided English translation.\n\nShloka:\n{shlok}\n\nTranslation:\n{translation}"
        reasoning_text = (
            "To build the glossary, I first need to divide the shloka into its semantic parts. "
            "By analyzing the given shloka and referencing the translation, I will break down the sandhi "
            "while keeping samasa (compound words) intact. "
            f"Upon dividing this, the resolved sequence of words is: [{sequence_string}]."
        )
        
        glossary_json = json.dumps({"glossary": clean_glossary}, ensure_ascii=False)
        output_1b = f"### REASONING:\n{reasoning_text}\n\n### FINAL GENERATION:\n```json\n{glossary_json}\n```"

        task_1b_PC.append({
            "prompt": [{"role": "user", "content": prompt_1b}],
            "completion": [{"role": "assistant", "content": output_1b}]
        })
        task_1b_messages.append({
            "messages": [
                {"role": "user", "content": prompt_1b},
                {"role": "assistant", "content": output_1b}
            ]
        })

    return task_1b_PC, task_1b_messages





if __name__=="__main__":
    dataset=load_dataset("sanganaka/padamitra")

    train_dataset=filter_sans_to_english(dataset["train"])
    eval_dataset=filter_sans_to_english(dataset["validation"])
    test_dataset=filter_sans_to_english(dataset["test"])

    print("TRAIN :", len(train_dataset))
    print("EVAL  :", len(eval_dataset))
    print("TEST  :", len(test_dataset))

    # print(train_dataset[0])
    train_task_1b_PC, train_task_1b_messages = format_instruct_json(train_dataset)
    eval_task_1b_PC, eval_task_1b_messages = format_instruct_json(eval_dataset)
    test_task_1b_PC, test_task_1b_messages = format_instruct_json(test_dataset)



    Dataset.from_list(train_task_1b_PC).save_to_disk("./data/pc_data_instruct_train_1b")
    Dataset.from_list(eval_task_1b_PC).save_to_disk("./data/pc_data_instruct_eval_1b")
    Dataset.from_list(test_task_1b_PC).save_to_disk("./data/pc_data_instruct_test_1b")
    
    Dataset.from_list(train_task_1b_messages).save_to_disk("./data/messages_data_instruct_train_1b")
    Dataset.from_list(eval_task_1b_messages).save_to_disk("./data/messages_data_instruct_eval_1b")
    Dataset.from_list(test_task_1b_messages).save_to_disk("./data/messages_data_instruct_test_1b")

    
