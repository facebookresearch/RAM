"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC-By-NC license found in the
LICENSE file in the root directory of this source tree.
"""

from sweet_rl.utils.webpage_utils import (
    extract_html_snippet,
    get_driver,
    render_full_html,
    replace_urls,
)
import json
from fire import Fire
import numpy as np
import concurrent
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torchvision.transforms import functional as F
import torch
import os
from tqdm import tqdm
import random


def main(
    saved_path="/fsx-ram/yifeizhou/collab_llm/outputs/temp_test.jsonl",
    k=1,
    temp_path="fsx-ram/yifeizhou/collab_llm/driver_cache/",
    preference_path=None,
    ground_truth_preference_path=None,
):
    with open(saved_path, "r") as fb:
        annotation_results = [json.loads(line) for line in fb]
    print(f"Number of trajectories: {len(annotation_results)}")
    num_tasks = len(annotation_results) // k
    for i in range(0, len(annotation_results) - num_tasks):
        assert (
            annotation_results[i]["task"]["problem_description"]
            == annotation_results[i + num_tasks]["task"]["problem_description"]
        )

    skip_evaluation = True
    for annotation_result in annotation_results:
        if not "reward" in annotation_result:
            skip_evaluation = False
            break
    skip_evaluation = False
    if not skip_evaluation:
        evaluation_batch_size = min(100, len(annotation_results))
        answer_images = [a["answer"] for a in annotation_results]
        ground_truth_images = [a["task"]["ground_truth"] for a in annotation_results]
        drivers = []
        print("Getting drivers")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs = [executor.submit(get_driver) for i in range(evaluation_batch_size)]
            drivers = [job.result() for job in jobs]
        print("Rendering images")
        rendered_images = []
        for i in tqdm(range(0, len(annotation_results), evaluation_batch_size)):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                jobs = [
                    executor.submit(
                        render_full_html,
                        driver,
                        ground_truth_images[i + j],
                        temp_path,
                        i + j,
                    )
                    for j, driver in enumerate(drivers)
                ]
                rendered_images += [job.result() for job in jobs]
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     jobs = [executor.submit(render_full_html, driver, ground_truth_images[i], "/fsx-ram/yifeizhou/collab_llm/driver_cache", i) for i, driver in enumerate(drivers)]
        #     rendered_images += [job.result() for job in jobs]
        for d in drivers:
            d.quit()
        ground_truth_images = [
            Image.open(ground_truth_image) for ground_truth_image in rendered_images
        ]
        answer_images = [
            (
                Image.open(answer_image).convert("RGB")
                if answer_image is not None and os.path.exists(answer_image)
                else Image.new("RGB", (224, 224), "black")
            )
            for answer_image in answer_images
        ]
        # import IPython; IPython.embed(); exit()
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        inputs1 = processor(images=answer_images, return_tensors="pt", padding=True).to(
            "cuda"
        )
        inputs2 = processor(
            images=ground_truth_images, return_tensors="pt", padding=True
        ).to("cuda")
        # Get the image embeddings
        with torch.no_grad():
            image_features1 = model.get_image_features(**inputs1)
            image_features2 = model.get_image_features(**inputs2)
        # Normalize the embeddings
        image_features1 = image_features1 / image_features1.norm(dim=-1, keepdim=True)
        image_features2 = image_features2 / image_features2.norm(dim=-1, keepdim=True)
        # Calculate cosine similarity
        similarities = (
            torch.sum(image_features1 * image_features2, dim=-1).cpu().numpy()
        )
        print(np.mean(similarities))
        for annotation_result, similarity in zip(annotation_results, similarities):
            annotation_result["reward"] = float(similarity)
        with open(saved_path, "w") as fb:
            for annotation_result in annotation_results:
                fb.write(json.dumps(annotation_result) + "\n")
    else:
        print("Skipping evaluation")
    similarities = np.array([a["reward"] for a in annotation_results])
    raw_correctness_results = similarities.reshape(k, -1)
    print(
        f"Best-of-{k} Average reward: {np.mean(np.max(raw_correctness_results, axis=0))}"
    )

    preference_pairs = []
    # best_correctness_indices = []
    if preference_path is not None or ground_truth_preference_path is not None:
        for i in range(num_tasks):
            #  if np.max(raw_correctness_results[:, i]) >= 1:
            best_correctness_index = (
                num_tasks * np.argmax(raw_correctness_results[:, i]) + i
            )
            # best_correctness_indices.append(best_correctness_index)
            # put failed trajectories to be rejected and the best trajectory to be accepted
            for j in range(k):
                for j2 in range(j + 1, k):
                    if (
                        abs(
                            raw_correctness_results[j, i]
                            - raw_correctness_results[j2, i]
                        )
                        < 0.1
                    ):
                        continue
                    if raw_correctness_results[j, i] > raw_correctness_results[j2, i]:
                        preference_pairs.append(
                            {
                                "chosen": annotation_results[num_tasks * j + i],
                                "rejected": annotation_results[num_tasks * j2 + i],
                            }
                        )
                    else:
                        preference_pairs.append(
                            {
                                "chosen": annotation_results[num_tasks * j2 + i],
                                "rejected": annotation_results[num_tasks * j + i],
                            }
                        )
                # if j != np.argmax(raw_correctness_results[:, i]) and raw_correctness_results[j, i] < np.max(raw_correctness_results[:, i]): #force a gap of 0.2 here
                #     preference_pairs.append({
                #         "chosen": annotation_results[best_correctness_index],
                #         "rejected": annotation_results[num_tasks*j + i],
                #     })
        print(f"Number of preference pairs: {len(preference_pairs)}")

        dummy_log = {
            "input": "The trajectory has ended",
            "output": "The trajectory has ended",
        }

        if ground_truth_preference_path is not None:
            with open(ground_truth_preference_path, "w") as fb:
                for preference_pair in preference_pairs:
                    chosen = preference_pair["chosen"]["dialogue_history"][-1]
                    ground_truth = preference_pair["chosen"]["task"]["ground_truth"]
                    chosen = {
                        "input": f"In light that the final answer is: {ground_truth}."
                        + chosen["input"],
                        "output": chosen["output"],
                    }
                    rejected = preference_pair["rejected"]["dialogue_history"][-1]
                    rejected = {
                        "input": f"In light that the final answer is: {ground_truth}."
                        + rejected["input"],
                        "output": rejected["output"],
                    }
                    fb.write(
                        json.dumps({"chosen": chosen, "rejected": rejected}) + "\n"
                    )
        else:
            with open(preference_path, "w") as fb:
                for preference_pair in preference_pairs:
                    chosen = preference_pair["chosen"]["dialogue_history"][-1]
                    rejected = preference_pair["rejected"]["dialogue_history"][-1]
                    chosen = preference_pair["chosen"]["dialogue_history"][-1]
                    ground_truth = preference_pair["chosen"]["task"]["ground_truth"]
                    chosen = {"input": chosen["input"], "output": chosen["output"]}
                    rejected = preference_pair["rejected"]["dialogue_history"][-1]
                    rejected = {
                        "input": rejected["input"],
                        "output": rejected["output"],
                    }
                    fb.write(
                        json.dumps({"chosen": chosen, "rejected": rejected}) + "\n"
                    )


if __name__ == "__main__":
    Fire(main)
