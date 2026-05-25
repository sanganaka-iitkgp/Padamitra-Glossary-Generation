import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, TrainingArguments, EarlyStoppingCallback, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset
from huggingface_hub import login
import os

login()

model_id="unsloth/phi-4"
train_dataset = Dataset.load_from_disk("../../data/data_instruct_train_1a")
eval_dataset = Dataset.load_from_disk("../../data/data_instruct_eval_1a")


torch.manual_seed(42)

processor = AutoTokenizer.from_pretrained(model_id)

processor.padding_side = "right"

local_rank = int(os.environ.get("LOCAL_RANK", 0))
device_map = {"": local_rank}

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.bfloat16, 
    bnb_4bit_use_double_quant=True
)

print(f"Loading model on GPU {local_rank} in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device_map,
    # device_map="auto",
    # device_map="cuda:1",
    quantization_config=bnb_config, 
    attn_implementation="sdpa"
)

peft_config = LoraConfig(
    r=256,            
    lora_alpha=16,        
    target_modules="all-linear",  
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    use_rslora=True     
)
# model=get_peft_model(model,peft_config)
# print(model.get_nb_trainable_parameters())
# print(processor.chat_template)


training_args = SFTConfig(
    report_to="wandb",        
    run_name="phi4-glossary-1a-v1",

    output_dir="./phi4-glossary-1a",
    per_device_train_batch_size=8, 
    gradient_accumulation_steps=2,
    
    optim="adamw_torch_fused", 
    logging_steps=10,
    
    save_strategy="steps",     
    save_steps=300,      ##160
    eval_strategy="steps",     
    eval_steps=300,

    bf16=True, 
    max_grad_norm=1.0, 

    learning_rate=3e-5, 
    warmup_ratio=0.15,
    lr_scheduler_type="cosine",
    
    seed=42, 

    num_train_epochs=10,               
    load_best_model_at_end=True,       
    metric_for_best_model="eval_loss", 
    greater_is_better=False,

    save_total_limit=4,
    max_length=512,
    
    # assistant_only_loss=True,
    completion_only_loss=True,
    
    ddp_find_unused_parameters=True,
    gradient_checkpointing=True
)

trainer = SFTTrainer( 
    model=model,                  
    train_dataset=train_dataset, 
    eval_dataset=eval_dataset,   
    args=training_args,
    peft_config=peft_config,
    processing_class=processor, 
    callbacks=[EarlyStoppingCallback(
        early_stopping_patience=8,
        early_stopping_threshold=0.001 
    )] 
)

trainer.train()

print("Saving the best model...")
trainer.save_model("./phi4-glossary-1a-best-model")
processor.save_pretrained("./phi4-glossary-1a-best-model")
print("Training complete and model saved!")