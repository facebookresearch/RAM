"""Output filenames shared by generation and the end-to-end CLI."""


def output_path(output, model, enable_nothink=False, temperature=0.6, num_gens=8):
    path = output.replace("{{model}}", model.replace("/", "-"))
    if enable_nothink:
        path = path.replace(".jsonl", "-nothink.jsonl")
    if temperature != 0.6:
        path = path.replace(".jsonl", f"-temp={temperature}.jsonl")
    if num_gens != 8:
        path = path.replace(".jsonl", f"-num_gens={num_gens}.jsonl")
    return path
