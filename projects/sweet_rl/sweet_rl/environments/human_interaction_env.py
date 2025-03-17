"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC-By-NC license found in the
LICENSE file in the root directory of this source tree.
"""

import openai

HUMAN_RESPONSE_CHARACTER_LIMIT = 400


class HumanInteractionEnv:
    def __init__(self, client, human_prompt, model_id, env_id=0, max_steps: int = 10):
        super().__init__()
        self.client = client
        self.human_prompt = human_prompt
        self.env_id = env_id

        self.model_id = model_id
        self.max_steps = max_steps
        self.problem_description = ""
        self.hidden_information = ""
        self.steps = 0
        self.dialogue_history = []
        self.done = False

    def get_dialogue_history(self):
        messages = [
            {
                "role": d["role"],
                "content": d["content"],
            }
            for d in self.dialogue_history
        ]
        return messages

    def str_dialogue_history(self):
        result = ""
        for d in self.dialogue_history:
            result += str(d["role"]) + ":"
            result += str(d["content"]) + "\n\n\n\n"
        return result + "agent:"

    # def str_dialogue_history(self):
    #     result = ''
    #     for d in self.dialogue_history:
    #         result += str(d["role"]) + ':\n'
    #         result += str(d["content"]) + '\n\n\n\n'
    #     return result + "agent:\n"

    def reset(self, problem_description, hidden_information):
        self.problem_description = str(problem_description)
        self.hidden_information = str(hidden_information)
        self.answer = "No answer"
        self.steps = 0
        self.done = False
        self.dialogue_history = []
        self.dialogue_history.append(
            {
                "role": "user",
                "content": problem_description,
            }
        )
        return self.get_dialogue_history()

    def invoke_model(self):
        for _ in range(3):
            try:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": self.human_prompt.format(
                            problem_description=self.problem_description,
                            hidden_information=self.hidden_information,
                            dialogue_history=self.str_dialogue_history(),
                        ),
                    },
                ]
                completion = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0,
                )
                return completion.choices[0].message.content
            except openai.BadRequestError as e:
                return "No response."
            # messages=[
            # {"role": "system", "content": "You are a helpful assistant."},
            # {"role": "user", "content": self.human_prompt.format(problem_description=self.problem_description,
            #                                                      hidden_information=self.hidden_information,
            #                                                      dialogue_history=self.str_dialogue_history())},
            # ]
            # completion = self.client.chat.completions.create(model=self.model_id, messages=messages, max_tokens=2048)
            # return completion.choices[0].message.content

    def step(self, response, formatted_prompt=None):
        self.steps += 1
        if self.done:
            return None, 0, True
        # if "human:" in response:
        #     response = response.split("human:")[0]
        raw_response = response

        if "OUTPUT:" in response:
            response = response.split("OUTPUT:")[1]
            # remove additional OUTPUT: if exists
            raw_response = "OUTPUT:".join(raw_response.split("OUTPUT:")[:2])

        if "I WANT TO ANSWER:" in response or self.steps >= self.max_steps:
            self.done = True
            if "I WANT TO ANSWER:" in response:
                self.answer = response.split("I WANT TO ANSWER:")[1]
            else:
                self.answer = response

        self.dialogue_history.append(
            {
                "role": "assistant",
                "content": response,
                "input": formatted_prompt,
                "output": raw_response,
            }
        )

        if not self.done:
            answer = self.invoke_model()
            self.dialogue_history.append(
                {"role": "user", "content": answer[:HUMAN_RESPONSE_CHARACTER_LIMIT]}
            )
        return self.get_dialogue_history() if not self.done else None, 0, self.done
