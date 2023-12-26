import os
import torch
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer
import json
import re
import sys
import multiprocessing

# Set up configuration
target_lang = sys.argv[1]
checkpoint_location = sys.argv[2]
checkpoint_n = int(sys.argv[3])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_threads = multiprocessing.cpu_count() if len(sys.argv) < 5 else int(sys.argv[4])

# Load the open assistant dataset
dataset = load_dataset("OpenAssistant/oasst1")

# Cache for loaded translation models, seemingly faster than letting Huggingface handle it
model_cache = {}

def load_model(model_name, model_key):
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
  model_cache[model_key] = (model, tokenizer)
  return model, tokenizer

# Tries to obtain a translation model from the Helsinki-NLP groups OPUS models. Returns None, None if no model is found for this language pair
def get_helsinki_nlp_model(source_lang, target_lang):
    model_key = f'{source_lang}-{target_lang}'

    if model_key in model_cache:
        return model_cache[model_key]

    model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    try:
      return load_model(model_name, model_key)
    except Exception as e:
      # Try to load the tc-big naming convention files
      try:
        model_name = f'Helsinki-NLP/opus-mt-tc-big-{source_lang}-{target_lang}'
        return load_model(model_name, model_key)
      except Exception as e:
        return None, None

# If a direct translation between two languages isn't possible, we ettempt to use English as a bridge (or any other intermediate lang)
def translate_text_through_english(text, source_lang, target_lang, intermediate_lang='en'):
    # Translate from source to English
    model, tokenizer = get_helsinki_nlp_model(source_lang, intermediate_lang)
    if model is None or tokenizer is None:
        return None
    inputs = tokenizer.encode(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        intermediate_translation = model.generate(inputs, max_length=512)
    text_in_english = tokenizer.decode(intermediate_translation[0], skip_special_tokens=True)

    # Translate from English to target
    model, tokenizer = get_helsinki_nlp_model(intermediate_lang, target_lang)
    if model is None or tokenizer is None:
        return None  # Can't perform translation
    inputs = tokenizer.encode(text_in_english, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        final_translation = model.generate(inputs, max_length=512)
    final_text = tokenizer.decode(final_translation[0], skip_special_tokens=True)

    return final_text

# Translate a given text from a source language into a target language
def translate_text(text, source_lang, target_lang):
    # Try direct translation first
    try:
      model, tokenizer = get_helsinki_nlp_model(source_lang, target_lang)
    except:
      model = None
      tokenizer = None
    if model is not None and tokenizer is not None:
        inputs = tokenizer.encode(text, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            translated = model.generate(inputs, max_length=512)
        return tokenizer.decode(translated[0], skip_special_tokens=True)

    # If direct translation is not available, use English as intermediate
    return translate_text_through_english(text, source_lang, target_lang)

# Find the max checkpoint number to continue from
def find_largest_checkpoint(checkpoint_location):
    pattern = r'upto_(\d+).json'
    files = os.listdir(checkpoint_location)
    numbers = [int(re.search(pattern, file).group(1)) for file in files if re.match(pattern, file)]
    if numbers:
        return max(numbers)
    else:
        return 0

# Processes records of a fold inside a thread
def process_records(records, fold, fold_checkpoint_location, target_lang, cnt_start):
    translated_texts = []
    for record in records:
        translated_text = translate_text(record['text'], record['lang'], target_lang) if record['lang'] != target_lang else record['text']

        new_record = {'text': translated_text, 'lang': target_lang if record['lang'] != target_lang else record['lang']}
        translated_texts.append(new_record)
    
    # Write out the processed records to a temporary file
    temp_file_path = f'{fold_checkpoint_location}/upto_{cnt_start}.json'
    with open(temp_file_path, 'w', encoding='utf-8') as f:
        json.dump(translated_texts, f)

    return temp_file_path

# Loop through the actual data and translate in parallel
for fold in dataset:
    records = list(dataset[fold])
    record_chunks = [records[i::num_threads] for i in range(num_threads)]

    fold_checkpoint_location = checkpoint_location + '/' + fold
    if not os.path.exists(fold_checkpoint_location):
        os.makedirs(fold_checkpoint_location)

    last_checkpoint_n = find_largest_checkpoint(fold_checkpoint_location)

    # Create a multiprocessing pool and process each chunk
    with multiprocessing.Pool(num_threads) as pool:
        results = pool.starmap(
            process_records,
            [(chunk, fold, fold_checkpoint_location, target_lang, last_checkpoint_n + len(chunk) * i) for i, chunk in enumerate(record_chunks)]
        )

    with open(f'{fold_checkpoint_location}/upto_{len(records)}.json', 'w', encoding='utf-8') as f:
        json.dump(final_translated_texts, f)