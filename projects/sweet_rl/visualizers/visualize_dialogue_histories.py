import gradio as gr
from PIL import Image
import torch
from fire import Fire
import json


def main(saved_path = "/fsx-ram/yifeizhou/collab_llm/outputs/temp_test.jsonl"):
    with open(saved_path, "r") as fb:
        annotation_results = [json.loads(line) for line in fb]
    # annotation_results = 

    print(len(annotation_results))
    # i = 0  # Initialize the index
    def update_label(i):
        loaded_result = annotation_results[i]["dialogue_history"]
        chatbot_results = []
        for j in range(0, len(loaded_result), 2):
            chatbot_results.append([loaded_result[j]["content"], loaded_result[j+1]["output"]])
        return chatbot_results, annotation_results[i]["answer"], annotation_results[i]["task"]["ground_truth"]
        # loaded_result = annotation_results[i]
        # return loaded_result["system_prompt"], loaded_result["raw_text"], loaded_result["result"]


    # Create the Gradio interface
    interface = gr.Interface(
        fn=update_label,
        inputs=[gr.components.Number()],
        outputs=[
            gr.components.Chatbot(),
            gr.components.Text(label="Answer"),
            gr.components.Text(label="Ground Truth"),
        ],
        title="Conversation Visualizer",
        description="Label the task from choices below and navigate through the dataset."
    )

    # Launch the interface
    interface.launch(share=True, server_port=9785)
    
if __name__ == "__main__":
    Fire(main)