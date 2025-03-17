from vllm import LLM, SamplingParams
from datasets import load_dataset
from huggingface_hub import login
import json
from transformers import AutoTokenizer
from openai import OpenAI
import concurrent
from fire import Fire
from tqdm import tqdm
import openai
from sweet_rl.environments import HumanInteractionEnv, HumanDesignInteractionEnv
from sweet_rl.models.vllm_agent import VLLMAgent
    
def batch_interact_environment(agent, environments, tasks):
    # import IPython; IPython.embed(); exit(1)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        jobs = [executor.submit(env.reset, task["problem_description"], task["ground_truth"]) for env, task in zip(environments, tasks)]
        batch_obs = [job.result() for job in jobs]
    for j in range(environments[0].max_steps+1): #just to be safe
        # print(f"Length of batch_obs: {len(batch_obs)}")
        formatted_prompts, responses = agent.get_action(batch_obs)
        # print(f"Length of formatted_prompts: {len(formatted_prompts)}")
        # print(f"Length of responses: {len(responses)}")
        # print(f"Length of environments: {len(environments)}")
        assert len(formatted_prompts) == len(responses)
        assert len(formatted_prompts) == len(environments)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs = [executor.submit(env.step, response, formatted_prompt) 
                    for env, response, formatted_prompt in 
                    zip(environments, responses, formatted_prompts)]
            batch_obs = [job.result()[0] for job in jobs]
    
    return [{
        "task": task,
        "dialogue_history":env.dialogue_history,
        "answer": env.answer} for task, env in zip(tasks, environments) ]
    
def main(hostname = "a100-st-p4de24xlarge-297",
    input_path = None, # path of the environment files
    output_path = "/fsx-ram/yifeizhou/collab_llm/outputs/temp_test.jsonl",
    user_prompt_path = "auto",
    agent_prompt_path = "auto",
    agent_model = "/fsx-ram/shared/Meta-Llama-3.1-8B-Instruct",
    env_model = "auto",
    batch_size = 1000, # parallel size for the environment
    num_tasks = 1000,
    max_steps = 10,
    best_of_n = 1, # n sampling per task
    train = False,
    task_type = "code", # code or html
    port = 8000, # port for environment servers
    temperature=1.0,
    temp_path = "/fsx-ram/yifeizhou/collab_llm/driver_cache",
    to_continue=False): #"/fsx-ram/shared/Meta-Llama-3.1-70B-Instruct",):
    if env_model == "auto":
        if task_type == "html":
            env_model = "/fsx-ram/shared/Qwen2-VL-72B-Instruct"
        else:
            env_model = "/fsx-ram/shared/Meta-Llama-3.1-70B-Instruct"
    if user_prompt_path == "auto":
        if task_type == "html":
            user_prompt_path = "prompts/human_simulator_html_prompt.txt"
        else:
            user_prompt_path = "prompts/human_simulator_code_prompt.txt"
    if agent_prompt_path == "auto":
        if task_type == "html":
            agent_prompt_path = "prompts/llm_agent_html_prompt.txt"
        else:
            agent_prompt_path = "prompts/llm_agent_code_prompt.txt"
    if task_type == "html":
        # very slow to have more than 100 driver instances running at the same time
        batch_size = 100
            
    tensor_parallel_size = torch.cuda.device_count()

    tokenizer = AutoTokenizer.from_pretrained(agent_model)
    
    with open(user_prompt_path, "r") as fb:
        human_prompt = fb.read()
    with open(agent_prompt_path, "r") as fb:
        agent_prompt = fb.read()
        
    with open(input_path, "r") as fb:
        tasks = [json.loads(line) for line in fb]
        tasks = tasks[:num_tasks]
    
    print(f"********************Number of tasks: {len(tasks)}**************")
    # clients of API servers for the environments
    clients = [OpenAI(base_url=f"http://{hostname}:{port}/v1", api_key="EMPTY") for _ in range(min(len(tasks)*best_of_n, batch_size))]
    
    print("Creating environments")
    if task_type == "html":
        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs = [executor.submit(HumanDesignInteractionEnv, client, human_prompt, env_model, i, max_steps, temp_path) for i, client in enumerate(clients)]
            environments = [job.result() for job in jobs]
    elif task_type == "code":
        environments = [HumanInteractionEnv(client = client,
                                        human_prompt = human_prompt,
                                        model_id = env_model,
                                        env_id = i,
                                        max_steps=max_steps) for i, client in enumerate(clients)]
    else:
        raise NotImplementedError
    agent = VLLMAgent(model_id=agent_model,
                agent_prompt=agent_prompt,
                tokenizer=tokenizer,
                tensor_parallel_size=tensor_parallel_size,
                temperature=temperature)

    trajectory = []
    # for _ in range(best_of_n):
    tasks = tasks*best_of_n
    print(f"Number of tasks: {len(tasks)}")
    if to_continue and output_path is not None:
        print("====================================Continuing from previous trajectory====================================")
        with open(output_path, "r") as fb:
            trajectory = [json.loads(line) for line in fb]
    for i in tqdm(range(len(trajectory), len(tasks), batch_size)):
        current_tasks = tasks[i:i+batch_size]
        # if not isinstance(agent, VLLMAgent):
        trajectory.extend(batch_interact_environment(agent, environments[:len(current_tasks)], current_tasks))
        with open(output_path, "w") as fb:
            for d in trajectory:
                fb.write(json.dumps(d) + "\n")
    
    # clean up the driver when done for the frontend environment
    if task_type == "html":
        for env in environments:
            env.driver.quit()

if __name__ == '__main__':
    Fire(main)
