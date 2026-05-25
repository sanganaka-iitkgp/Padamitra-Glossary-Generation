import datasets
from datasets import load_dataset, Dataset
from huggingface_hub import login
import json
import re

login()

def filter_sans_to_english(dataset):
    filtered_dataset=dataset.filter(lambda example: example['source_language']=='sanskrit' and example['target_language']=='english')
    return filtered_dataset

def parse_dirty_glossary(dirty_str):
    clean_str = re.sub(r'\},?\n?\}?$', '', str(dirty_str))
    clean_str = clean_str.strip('{ \n\r')
    
    glossary_dict = {}
    
    pairs = clean_str.split(',\n')
    
    for pair in pairs:
        if ':' in pair:
            key, val = pair.split(':', 1) 
            glossary_dict[key.strip()] = val.strip()
            
    return glossary_dict

def format_instruct_json(dataset):
    task_1a_data = []

    for row in dataset:
        shlok = row['shlok']
        translation = row['translation']
        clean_glossary = parse_dirty_glossary(row['glossary'])
        resolved_sequence = list(clean_glossary.keys())
        if resolved_sequence is None:
            print("Oh my habibiii")
        
        prompt_1a = f"Generate a Sanskrit-English glossary mapping for the following shloka based on its translation.\n\nShloka:\n{shlok}\n\nTranslation:\n{translation}"
        completion_1a = json.dumps({"glossary": clean_glossary}, ensure_ascii=False)
        
        task_1a_data.append({
            "messages": [
                {"role": "user", "content": prompt_1a},
                {"role": "assistant", "content": completion_1a}
            ]
        })


    return task_1a_data

if __name__=="__main__":
    dataset=load_dataset("sanganaka/padamitra")

    train_dataset=filter_sans_to_english(dataset["train"])
    eval_dataset=filter_sans_to_english(dataset["validation"])
    test_dataset=filter_sans_to_english(dataset["test"])

    print("TRAIN :", len(train_dataset))
    print("EVAL  :", len(eval_dataset))
    print("TEST  :", len(test_dataset))

    print(train_dataset[0])
    train_instruct_1a = format_instruct_json(train_dataset)
    eval_instruct_1a = format_instruct_json(eval_dataset)
    test_instruct_1a = format_instruct_json(test_dataset)
    
    Dataset.from_list(train_instruct_1a).save_to_disk("./data/qwen_data_instruct_train_1a")
    Dataset.from_list(eval_instruct_1a).save_to_disk("./data/qwen_data_instruct_eval_1a")
    Dataset.from_list(test_instruct_1a).save_to_disk("./data/qwen_data_instruct_test_1a")
    