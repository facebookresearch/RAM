from .human_interaction_env import HumanInteractionEnv
from sweet_rl.utils.webpage_utils import extract_html_snippet, get_driver, render_full_html
import openai

HUMAN_RESPONSE_CHARACTER_LIMIT = 1000


import base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return f"data:image;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"
    

class HumanDesignInteractionEnv(HumanInteractionEnv):
    def __init__(self, client, 
                 human_prompt,
                 model_id,
                 env_id = 0,
                 max_steps: int = 10,
                 temp_path = "~/.cache"):
        super().__init__(
            client = client,
            human_prompt = human_prompt,
            model_id = model_id,
            env_id = env_id,
            max_steps = max_steps
        )
        self.driver = get_driver()
        self.temp_path = temp_path
        
    def str_dialogue_history(self):
        result = ''
        for d in self.dialogue_history:
            result += str(d["role"]) + ':'
            result += str(d["content"]) + '\n\n\n\n'
        return result
    
    def get_dialogue_history(self):
        messages = [{
            "role": d["role"],
            "content": d["content"],
        } for d in self.dialogue_history]
        return messages
    
    def reset(self, problem_description, hidden_information):
        self.problem_description = str(problem_description)
        self.hidden_information = str(hidden_information)
        self.ground_truth_design = render_full_html(self.driver, self.hidden_information, self.temp_path, self.env_id)
        if self.ground_truth_design is None:
            self.b64_ground_truth_design = None
            self.done = True
        else:
            self.b64_ground_truth_design = encode_image(self.ground_truth_design)
        self.answer = "No answer"
        self.steps = 0
        self.done = False
        self.dialogue_history = []
        self.dialogue_history.append({
            "role": "user",
            "content": problem_description,
        })
        return self.get_dialogue_history()
        # return self.str_dialogue_history()
        
    def invoke_model(self, agent_output, agent_image=None):
        for _ in range(3):
            try:
                user_message = [{ "type": "text", "text": self.human_prompt},] #.format(problem_description=self.problem_description, agent_output=agent_output)},]
                if agent_image is not None:
                    user_message.append({"type": "text", "text": "Below is the design the agent is referring to."})
                    user_message.append({'type': 'image_url', 'image_url': {"url": encode_image(agent_image)}})
                else:
                    user_message.append({"type": "text", "text": "The agent did not provide any visualization."})

                user_message.append({"type": "text", "text": "Below is the ground truth design that the human user wants."})
                user_message.append({'type': 'image_url', 'image_url': {"url": self.b64_ground_truth_design}})
                    
                messages=[
                {"role": "user", "content": user_message},
                ]
                completion = self.client.chat.completions.create(model=self.model_id, messages=messages, max_tokens=4096, temperature=0)
                return completion.choices[0].message.content
            except openai.BadRequestError as e:
                print("Bad request error, retrying...")
                return "No response."
            # except openai.InternalServerError as e:
            #     print("Internal server error, retrying...")
            #     return "No response."
    
    def step(self, response, formatted_prompt = None):
        self.steps += 1
        if self.b64_ground_truth_design is None:
            self.done = True
        if self.done:
            return None, 0, True
        raw_response = response
        
        # DO NOT HAVE THIS THINKING STEP
        if "OUTPUT:" in response:
            response = response.split("OUTPUT:")[1]
            #remove additional OUTPUT: if exists
            raw_response = "OUTPUT:".join(raw_response.split("OUTPUT:")[:2])
        
        # DO NOT TRUNCATE THE HTML CODE
        _, agent_html = extract_html_snippet(response)
        
        if agent_html is not None:
            agent_image = render_full_html(self.driver, agent_html, self.temp_path, self.env_id)
            self.answer = agent_image
        else:
            agent_image = None
        
        if self.steps >= self.max_steps or "I WANT TO ANSWER:" in response:
            self.done = True
            
        self.dialogue_history.append({
            "role": "assistant",
            "content": response,
            "input": formatted_prompt,
            "output": raw_response,
        })
            
        if not self.done:
            answer = self.invoke_model(response, agent_image)
            self.dialogue_history.append(
                {
                    "role": "user",
                    "content": answer[:HUMAN_RESPONSE_CHARACTER_LIMIT]
                }
            )
        return self.get_dialogue_history() if not self.done else None, 0, self.done
        # return self.str_dialogue_history() if not self.done else None, 0, self.done
    