import gradio as gr
from PIL import Image
import torch
from fire import Fire
import json
from sweet_rl.utils.webpage_utils import extract_html_snippet, get_driver, render_full_html, replace_urls

def main(saved_path = "/fsx-ram/yifeizhou/collab_llm/outputs/temp_test.jsonl", temp_path = "/fsx-ram/yifeizhou/collab_llm/driver_cache"):
    driver = get_driver()
    
    with open(saved_path, "r") as fb:
        annotation_results = [json.loads(line) for line in fb]
    # annotation_results =  d

    print(len(annotation_results))
    # i = 0  # Initialize the index
    def update_label(i):
        loaded_result = annotation_results[i]["dialogue_history"]
        # visualized_contents = []
        # for d in loaded_result:
        #     if 
        chatbot_results = []
        for j in range(0, len(loaded_result), 2):
            # chatbot_results.append([loaded_result[j]["content"], loaded_result[j+1]["output"]])
            chatbot_results.append({"role": "user", "content": loaded_result[j]["content"]})
            response = loaded_result[j+1]["output"]
            if "OUTPUT:" in response:
                response = response.split("OUTPUT:")[1]
            output, html_snippet = extract_html_snippet(response)
            chatbot_results.append({"role": "assistant", "content": output})
            if html_snippet is not None:
                html_image = render_full_html(driver, html_snippet, temp_path)
                if html_image is not None:
                    chatbot_results.append({"role": "assistant", "content": gr.Image(html_image)})
            # else:
            #     html_image = None
            # chatbot_results.append({"role": "assistant", "content": output)
            # if html_image is not None:
                
            # chatbot_results.append([loaded_result[j]["content"], loaded_result[j+1]["output"]])
        return chatbot_results, gr.Image(annotation_results[i]["answer"]), render_full_html(driver, annotation_results[i]["task"]["ground_truth"], temp_path)
        # loaded_result = annotation_results[i]
        # return loaded_result["system_prompt"], loaded_result["raw_text"], loaded_result["result"]


    # Create the Gradio interface
    interface = gr.Interface(
        fn=update_label,
        inputs=[gr.components.Number()],
        outputs=[
            gr.components.Chatbot(type="messages"),
            gr.Image(label="Answer"),
            gr.Image(label="Ground Truth"),
        ],
        title="Conversation Visualizer",
        description="Label the task from choices below and navigate through the dataset."
    )

    # Launch the interface
    interface.launch(share=True, server_port=9785, allowed_paths=[temp_path])
    
if __name__ == "__main__":
    Fire(main)