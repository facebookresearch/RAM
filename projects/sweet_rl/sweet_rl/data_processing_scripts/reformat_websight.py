from sweet_rl.utils.webpage_utils import replace_urls, render_full_html
import json
from tqdm import tqdm
tasks_output_path = "/fsx-ram/yifeizhou/collab_llm/outputs/webpage_tasks_all.jsonl"

from datasets import load_dataset

ds = load_dataset("HuggingFaceM4/WebSight", "v0.2")["train"]


filtered_data = []
for i in tqdm(range(20000)):
    filtered_data.append({
        "problem_description": ds[i]["llm_generated_idea"], 
        "ground_truth": replace_urls(ds[i]["text"]),
    })

with open(tasks_output_path, "w") as f:
    for d in filtered_data:
        f.write(json.dumps(d) + "\n")