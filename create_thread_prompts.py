from datasets import load_dataset, DatasetDict, Dataset
import sys
import os
import pandas as pd
from tqdm import tqdm

dataset_name = sys.argv[1]
system_instruction = sys.argv[2]
output_location = sys.argv[3]

if os.path.isdir(dataset_name):
    dataset = load_dataset('arrow', data_files=os.path.join(dataset_name, '*.arrow'))
else:
    dataset = load_dataset(dataset_name)

# Function to format a thread in LLaMa2 format
def format_thread(thread):
    formatted_thread = f"<s>[INST] <<SYS>>\n{system_instruction}\n<</SYS>>\n\n"

    for i in range(0, len(thread), 2):
        user_msg = thread[i]['text'] if i < len(thread) else ""
        model_answer = thread[i+1]['text'] if i+1 < len(thread) else ""
        formatted_thread += f"{user_msg} [/INST] {model_answer} </s>"
        if i+2 < len(thread):
            formatted_thread += f"<s>[INST] "

    return formatted_thread

# We only continue the thread with the highest ranked answer to each input
def find_highest_ranked_child(parent_id):
      children = df[df['parent_id'] == parent_id]
      if not children.empty:
          return children.loc[children['rank'].idxmax()]
      return None

# Construct threads
folds = dataset.keys()
threads = {k: [] for k in folds}
for fold in folds:
  df = dataset[fold].to_pandas()

  # Replace NULLs in 'rank' with a value lower than the lowest rank
  min_rank = df['rank'].min()
  df['rank'].fillna(min_rank - 1, inplace=True)

  # Identify root messages (those without a parent_id)
  root_messages = df[df['parent_id'].isna()]

  with tqdm(total=root_messages.size) as pbar:
      for _, root_message in root_messages.iterrows():
          # Create the thread
          thread = [{'text': root_message['text']}]
          next_message = find_highest_ranked_child(root_message['message_id'])
    
          while next_message is not None:
              thread.append({'text': next_message['text']})
              next_message = find_highest_ranked_child(next_message['message_id'])
    
          # Turn this into LLaMa2 format
          threads[fold].append(format_thread(thread))
          # Update progress
          pbar.update(1)

  threads[fold] = Dataset.from_pandas(pd.DataFrame(data=threads[fold]))

dataset = DatasetDict(threads)
for fold in folds:
    dataset[fold] = dataset[fold].rename_column('0', 'text')

# Check if output location is a valid directory
if os.path.isdir(output_location):
    dataset.save_to_disk(output_location)
else:
    # Try to push to hub, requires HF_TOKEN environment variable to be set, see https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hftoken
    dataset.push_to_hub(output_location)
