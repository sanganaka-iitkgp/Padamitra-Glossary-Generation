import datasets
from datasets import load_dataset, Dataset
from huggingface_hub import login
import json
import re
import ast
import random

login()


def jumble_shloka(shloka: str, seed: int = None) -> str:
    if seed is not None:
        random.seed(seed)


    parts = []
    delimiters = []

    remaining = shloka
    while remaining:
        idx_double = remaining.find("।।")
        idx_single = remaining.find("।")

        if idx_double == -1 and idx_single == -1:
            parts.append(remaining)
            break

        if idx_double != -1 and (idx_single == -1 or idx_double <= idx_single):
            parts.append(remaining[:idx_double])
            delimiters.append("।।")
            remaining = remaining[idx_double + 2:]
        else:
            parts.append(remaining[:idx_single])
            delimiters.append("।")
            remaining = remaining[idx_single + 1:]

    jumbled_parts = []
    for part in parts:
        stripped = part.strip()
        if not stripped:
            jumbled_parts.append(part)
            continue
        words = stripped.split()
        random.shuffle(words)
        leading  = part[: len(part) - len(part.lstrip())]
        trailing = part[len(part.rstrip()):]
        jumbled_parts.append(leading + " ".join(words) + trailing)

    result = ""
    for i, part in enumerate(jumbled_parts):
        result += part
        if i < len(delimiters):
            result += delimiters[i]

    return result



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



def format_instruct_1a_jumbled(dataset):
    task_1a_data = []

    for row in dataset:
        seed = hash(row['shlok']) % (2**32)
        shlok       = jumble_shloka(row['shlok'], seed=seed) 
        translation = row['translation']
        clean_glossary = parse_dirty_glossary(row['glossary'])

        prompt_1a = (
            f"Generate a Sanskrit-English glossary mapping for the following shloka "
            f"based on its translation.\n\nShloka:\n{shlok}\n\nTranslation:\n{translation}"
        )
        completion_1a = json.dumps({"glossary": clean_glossary}, ensure_ascii=False)

        task_1a_data.append({
            "prompt":     [{"role": "user",      "content": prompt_1a}],
            "completion": [{"role": "assistant",  "content": completion_1a}]
        })
    return task_1a_data



def format_instruct_1b_jumbled(dataset):
    task_1b_PC       = []
    task_1b_messages = []

    for row in dataset:
        seed = hash(row['shlok']) % (2**32)
        shlok       = jumble_shloka(row['shlok'], seed=seed) 
        translation    = row['translation']
        clean_glossary = parse_dirty_glossary(row['glossary'])
        resolved_sequence = list(clean_glossary.keys())
        if not resolved_sequence:
            print("Oh my habibiii")

        sequence_string = ", ".join(resolved_sequence)

        prompt_1b = (
            f"### INPUT:\nYou are an expert in Sanskrit. Perform Padaccheda (word resolution) "
            f"on the following Sanskrit shloka, then generate the glossary mapping using the "
            f"generated Padaccheda and provided English translation.\n\nShloka:\n{shlok}"
            f"\n\nTranslation:\n{translation}"
        )

        reasoning_text = (
            "To build the glossary, I first need to divide the shloka into its semantic parts. "
            "By analyzing the given shloka and referencing the translation, I will break down the sandhi "
            "while keeping samasa (compound words) intact. "
            f"Upon dividing this, the resolved sequence of words is: [{sequence_string}]."
        )

        glossary_json = json.dumps({"glossary": clean_glossary}, ensure_ascii=False)
        output_1b = (
            f"### REASONING:\n{reasoning_text}\n\n"
            f"### FINAL GENERATION:\n```json\n{glossary_json}\n```"
        )

        task_1b_PC.append({
            "prompt":     [{"role": "user",      "content": prompt_1b}],
            "completion": [{"role": "assistant",  "content": output_1b}]
        })
        task_1b_messages.append({
            "messages": [
                {"role": "user",      "content": prompt_1b},
                {"role": "assistant", "content": output_1b}
            ]
        })

    return task_1b_PC, task_1b_messages



if __name__ == "__main__":

    dataset = load_dataset("sanganaka/padamitra")

    test_dataset  = filter_sans_to_english(dataset["test"])

    print("TEST  :", len(test_dataset))



    sample = test_dataset[0]['shlok']
    print("\nOriginal shloka :", sample)
    print("Jumbled  shloka :", jumble_shloka(sample))



    print("\nBuilding 1a jumbled datasets...")
    test_1a  = format_instruct_1a_jumbled(test_dataset)

    Dataset.from_list(test_1a ).save_to_disk("./jumbled_data/jumbled_data_instruct_test_1a")
    print("1a datasets saved.")



    print("\nBuilding 1b jumbled datasets...")
    test_1b_PC,  test_1b_msg  = format_instruct_1b_jumbled(test_dataset)

    Dataset.from_list(test_1b_PC  ).save_to_disk("./jumbled_data/jumbled_pc_data_instruct_test_1b")
    print("1b datasets saved.")

    print("\nAll jumbled datasets saved to ./jumbled_data/")