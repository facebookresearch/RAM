"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

QUESTION_FORMATS = {
    "numeric": """Additionally answer to every question should be a numeric value, that can be matched to gold answer. However, the answer to the question should be clear. Answer in similar json format.
    
## Important: Make sure a.) Questions have 1 clear numerical answer with no ambiguity or any potential similar or nearby answer.
For example, a bad question would be: "How many people are affected by diabetes worldwide in millions?" This is a bad question, because the number keeps on changing every year, is based on estimate and therefore answers could vary. Do not output such questions.
""",
    "mcq": """Additionally every question will be MCQ, with only one correct option and total of {NUM_OPTIONS} options, of which clearly 1 is correct without ambiguity. Answer in similar json format.""",
    "open-ended": """Additionally answer to every question should be a short answer such as single word or phrase, that can be matched to gold answer. However, the answer to the question should be clear. Answer in similar json format.""",
    "open-ended-long": """Additionally answer to every question should be a long answer such as a paragraph, that will be judged by a separte LLM as judge against the reference answer. However, the answer to the question should be clear without ambiguity. Answer in similar json format.""",
}

OVERTHINKING_PROMPT = """Suppose I have this problem. So basically there are these recent models that are called reasoning models, and the idea is that if you increase the inference compute, in the sense that if they generate longer chain of thoughts, the accuracy increases. However, one big challenge with them is that they sometimes overthink, spending a lot of compute even on simple questions, results in reduced utility for user, as it takes a lot of time generating long thinking. A simple question could be 2+2, and the model is expected to answer immediately.

In order to evaluate this behavior I plan to propose OverthinkingBench. THe idea is simple, this benchmark would contain some very simple questions, where model is not expected to think for more than 10 to 20 tokens, and sometimes 0 tokens to answer them. The accuracy would mostly be 100\% because the questions would be simple, and our evaluation would be average tokens used, standard deviation, or thinking violations (how many time thinking was > 20tokens).

Now I want you to make prompts for such a dataset. Note, that distribution of prompts should be similar to standard benchmarks for Large language models. Simple questions, although of varying difficulty, varying domains, varying types. Your goal is to create 50 such prompts. Diversity along different dimensions is expected.

OUTPUT FORMAT: output json, List[dict], where dict contains two keys: "question", "answer", "domain"

Domains: I will give you the following domains, and you are expected to generate some simple (not at all tough). This is to ensure the benchmark contains real world queries. The question can be straightforward factual questions, have some very basic multi-hop reasoning required, or some very basic math questions.

{{question_format}}

Here are the domains:
{{domains}}

For each domain create 15 questions.
"""

ALL_DOMAINS = [
    "Electronic Science and Technology",
    "Philosophy",
    "Traditional Chinese Medicine",
    "Applied Economics",
    "Mathematics",
    "Physics",
    "Clinical Medicine",
    "Computer Science and Technology",
    "Information and Communication Engineering",
    "Control Science and Engineering",
    "Theoretical Economics",
    "Law",
    "History",
    "Basic Medicine",
    "Education",
    "Materials Science and Engineering",
    "Electrical Engineering",
    "Systems Science",
    "Power Engineering and Engineering Thermophysics",
    "Military Science",
    "Biology",
    "Business Administration",
    "Language and Literature",
    "Public Health and Preventive Medicine",
    "Political Science",
    "Chemistry",
    "Hydraulic Engineering",
    "Chemical Engineering and Technology",
    "Pharmacy",
    "Geography",
    "Art Studies",
    "Architecture",
    "Forestry Engineering",
    "Public Administration",
    "Oceanography",
    "Journalism and Communication",
    "Nuclear Science and Technology",
    "Weapon Science and Technology",
    "Naval Architecture and Ocean Engineering",
    "Environmental Science and Engineering",
    "Transportation Engineering",
    "Geology",
    "Physical Oceanography",
    "Musicology",
    "Stomatology",
    "Aquaculture",
    "Mechanical Engineering",
    "Aeronautical and Astronautical Science and Technology",
    "Civil Engineering",
    "Mechanics",
    "Petroleum and Natural Gas Engineering",
    "Sociology",
    "Food Science and Engineering",
    "Agricultural Engineering",
    "Surveying and Mapping Science and Technology",
    "Metallurgical Engineering",
    "Library, Information and Archival Management",
    "Mining Engineering",
    "Astronomy",
    "Geological Resources and Geological Engineering",
    "Atmospheric Science",
    "Optical Engineering",
    "Animal Husbandry",
    "Geophysics",
    "Crop Science",
    "Management Science and Engineering",
    "Psychology",
    "Forestry",
    "Textile Science and Engineering",
    "Veterinary Medicine",
    "Instrument Science and Technology",
    "Physical Education",
]
