"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC-By-NC license found in the
LICENSE file in the root directory of this source tree.
"""

import multiprocessing
from queue import Empty

import signal

disable_printing = """
import sys
class DisablePrint:
    def write(self, x):
        pass
    def flush(self):
        pass
# Save the current state of stdout
original_stdout = sys.stdout
# Disable all printing to the console
sys.stdout = DisablePrint()
"""


# Handler function to be called when the alarm signal is received
def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")


def get_function_output(function_definition, test_case):
    # exec(disable_printing)
    try:
        exec(function_definition)
        return eval(test_case)
    except Exception as e:
        return None


def queue_get_function_output(function_definition, test_case, queue):
    queue.put(get_function_output(function_definition, test_case))


def subprocess_get_function_output(function_definition, test_case):
    # do not want any os functions
    if (
        "import os" in function_definition
        or "from os" in function_definition
        or "import sys" in function_definition
        or "from sys" in function_definition
    ):
        return None
    if (
        "open(" in function_definition
        or "print(" in function_definition
        or "write" in function_definition
    ):
        return None
    if "sudo" in function_definition or "transformers" in function_definition:
        return None
    if "exit(" in function_definition or "quit(" in function_definition:
        return None

    # Set the signal handler for SIGALRM
    # signal.signal(signal.SIGALRM, timeout_handler)
    # signal.alarm(1)  # Set an alarm for 10 seconds
    # try:
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=queue_get_function_output, args=(function_definition, test_case, queue)
    )
    process.start()
    process.join(timeout=1)
    process.kill()
    # except TimeoutError:
    #     return None
    # process.terminate()
    try:
        result = queue.get(timeout=0.1)
    except Empty:
        result = None
    return result


def check_correctness(ground_truth_function, test_function, test_cases):
    # Although unlikely, there is a chance that this function may run malicious code outputted by the LLMs
    num_correct = 0

    if (
        "import os" in test_function
        or "from os" in test_function
        or "import sys" in test_function
        or "from sys" in test_function
    ):
        return 0
    if "sudo" in test_function or "transformers" in test_function:
        return 0
    if "exit(" in test_function or "quit(" in test_function:
        return 0
    if "argparse" in test_function:
        return 0

    if "```python" in test_function:
        test_function = test_function.split("```python")[1].split("```")[0]
    if "```" in test_function:
        test_function = test_function.split("```")[1].split("```")[0]

    for test_case in test_cases.values():
        ground_truth_output = get_function_output(ground_truth_function, test_case)

        # timeout precautions
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(1)  # Set an alarm for 10 seconds
        try:
            # print(test_function)
            # if "match_player" in test_function:
            #     return 0
            test_output = get_function_output(test_function, test_case)
        except TimeoutError:
            test_output = None
        signal.alarm(0)  # Reset the alarm
        try:
            if ground_truth_output == test_output and ground_truth_output is not None:
                num_correct += 1
        except ValueError:
            pass
    return num_correct / len(test_cases)


def code_evaluate(trajectories):
    all_correctness = []
    for i, trajectory in enumerate(trajectories):
        # if i == 15051 or i == 15276:
        #     all_correctness.append(0)
        #     continue
        # print(i)
        ground_truth_function = trajectory["task"]["ground_truth"]
        test_function = trajectory["answer"]
        # print(test_function)
        test_cases = trajectory["task"]["test_cases"]
        correctness = check_correctness(
            ground_truth_function, test_function, test_cases
        )
        # if correctness < 1:
        #     print(i)
        #     print(test_function)
        all_correctness.append(correctness)
    print(f"Average correctness: {sum(all_correctness)/len(all_correctness)}")
    print(f"Number of trajectories: {len(all_correctness)}")
    print(
        f"Percentage of correct trajectories: {sum([1 for correctness in all_correctness if correctness == 1])/len(all_correctness)}"
    )
    return all_correctness
