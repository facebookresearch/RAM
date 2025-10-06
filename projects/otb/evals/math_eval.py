from math_verify import parse, verify


def eval_math(row, tokenizer=None, model_name: str | None = None):
    scores = []
    gt = row['answer']
    if '\\boxed{' not in gt:
        gt = '\\boxed{' + gt + '}'
    gt = parse(gt)
    responses = row['response'] if isinstance(row['response'], list) else [row['response']]
    for res in responses:
        res = parse(res)
        scores.append(verify(gt, res))
    return sum(scores) / len(scores) if len(scores) > 0 else 0.0

