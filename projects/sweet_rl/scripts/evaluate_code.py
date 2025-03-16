import json
from fire import Fire
from sweet_rl.utils import code_evaluate
import numpy as np
import random

def main(saved_path = "/fsx-ram/yifeizhou/collab_llm/outputs/temp_test.jsonl",
         k = 1, #how many times each task has been repeated
         preference_path = None, # the path to save trajectory-level preferences
         ground_truth_preference_path = None, # the path to save trajectory-level preferenecs with additional training time information (ground truth)
         ):
    with open(saved_path, "r") as fb:
        annotation_results = [json.loads(line) for line in fb]
    print(f"Number of trajectories: {len(annotation_results)}")
    num_tasks = len(annotation_results)//k
    for i in range(0, len(annotation_results) - num_tasks):
        assert annotation_results[i]["task"]["problem_description"] == annotation_results[i + num_tasks]["task"]["problem_description"]
    # annotation_results = annotation_results[15000:]

    need_reevaluate = False
    for a in annotation_results:
        if not "reward" in a:
            need_reevaluate = True
            break
    
    if need_reevaluate:
        raw_correctness_results = np.array(code_evaluate(annotation_results)).reshape(k, -1)
        print("=====>Saving evaluation results")
        with open(saved_path, "w") as fb:
            for a in annotation_results:
                fb.write(json.dumps(a) + "\n")
    else:
        print("Using existing correctness results")
        raw_correctness_results = np.array([a["reward"] for a in annotation_results]).reshape(k, -1)
        
    correctness_results = np.max(raw_correctness_results, axis=0)
    print(f"Best-of-{k} Average correctness: {np.mean(correctness_results)}")
    print(f"Best-of-{k} percentage of correct trajectories: {np.sum(correctness_results == 1)/len(correctness_results)}")
    # save the evaluated rewards
    for a, r in zip(annotation_results, raw_correctness_results.reshape(-1).tolist()):
        a["reward"] = r
    
    best_correctness_indices = []
    preference_pairs = []
    if preference_path is not None or ground_truth_preference_path is not None:
        for i in range(num_tasks):
             if np.max(raw_correctness_results[:, i]) >= 1:
                best_correctness_index = num_tasks * np.argmax(raw_correctness_results[:, i]) + i
                best_correctness_indices.append(best_correctness_index)
                # put failed trajectories to be rejected and the best trajectory to be accepted
                for j in range(k):
                    if j != np.argmax(raw_correctness_results[:, i]) and raw_correctness_results[j, i] < np.max(raw_correctness_results[:, i]):
                        preference_pairs.append({
                            "chosen": annotation_results[best_correctness_index],
                            "rejected": annotation_results[num_tasks*j + i],
                        })
        print(f"Number of preference pairs: {len(preference_pairs)}")
        

        if ground_truth_preference_path is not None:
            with open(ground_truth_preference_path, "w") as fb:
                for preference_pair in preference_pairs:
                    chosen = preference_pair["chosen"]["dialogue_history"][-1]
                    ground_truth = preference_pair["chosen"]["task"]["ground_truth"]
                    chosen = f"In light that the final answer is: {ground_truth}."+chosen["input"]+chosen["output"]
                    rejected = preference_pair["rejected"]["dialogue_history"][-1]
                    rejected = f"In light that the final answer is: {ground_truth}."+rejected["input"] + rejected["output"]
                    fb.write(json.dumps({
                        "chosen": chosen,
                        "rejected": rejected
                    }) + "\n")
        else:
            with open(preference_path, "w") as fb:
                for preference_pair in preference_pairs:
                    chosen = preference_pair["chosen"]["dialogue_history"][-1]
                    rejected = preference_pair["rejected"]["dialogue_history"][-1]
                    chosen = preference_pair["chosen"]["dialogue_history"][-1]
                    ground_truth = preference_pair["chosen"]["task"]["ground_truth"]
                    chosen = chosen["input"] + chosen["output"]
                    # {
                    #     "input": chosen["input"],
                    #     "output": chosen["output"]
                    # }
                    rejected = preference_pair["rejected"]["dialogue_history"][-1]
                    rejected = rejected["input"] + rejected["output"]
                    # {
                    #     "input": rejected["input"],
                    #     "output": rejected["output"]
                    # }
                    fb.write(json.dumps({
                        "chosen": chosen,
                        "rejected": rejected
                    }) + "\n")
        
if __name__ == "__main__":
    Fire(main)