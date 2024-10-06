OPENAI_API_KEY_0 ='sk-'
OPENAI_API_KEY_1 ='sk-'
OPENAI_API_KEY_2 ='sk-'
OPENAI_API_KEY_3 ='sk-'
OPENAI_API_KEY = OPENAI_API_KEY_2

OPENAI_BASE_URL =''

MODEL_TYPE = "gpt-4o"

PROMPT_FOR_TEXT_MATCHING='''
I will provide you with the contents of the feature and the truth. 
You need to help me determine whether these two pieces of information are talking about the same thing and score their relevance. 
The score can be either 0 or 1. 
The output format is: Score: [01](regular expression)
The output format must strictly adhere to the following regular expression: "Score: [01]"
'''
PROMPT_FOR_LOGIC_CHECKING='''
You are a specialized scoring agent. 
I will provide you with some basic information and the system's output.
You need to judge whether the system's output is logically reasonable based on the basic information and score it from zero to five.
The output format is: r"Score:\s*([0-5])"(regular expression)
The output format must strictly adhere to the following regular expression: r"Score:\s*([0-5])"
'''

PROMPT_FOR_KEY_INFOMATION_CHECKING = '''
You are now an agent specializing in scoring.
I will give you several key pieces of information separated by commas and a segment of system output. 
You need to help me check how many of the key pieces of information are included in the system output. 
The output content should tell me how many key pieces of information (different key pieces of information will be separated by commas)
and tell me how many key pieces of information are included in the system output.

The output format must strictly follow the following form: Key Information Count: \d+, Included Key Information Count: \d+

for example:
the input is: key information: ["mask thief gone","drive vehicle to steal","windows damaging by thief","got property damaging"] system output: "The video shows a news segment where a man driving vehicle tried to steal a bag" 
the out put should be: Key Information Count: 4, Included Key Information Count: 1
'''
 
PROMPTS_FOR_REASONING_ABILITY_CHECKING='''
We are doing experiments to see if multimodal large models have inference capabilities. 
The way we do this is we feed a multimodal large model a video without any information and let it understand and reason. 
Normally it should not be able to make any judgments and tell us that it can't or that the video doesn't provide meaningful content, 
but sometimes it will reason based on hallucinations. You need to help us determine whether it's output something like "I can't infer" or "the video doesn't provide meaningful content" or "None" to prove that it's actually speaking based on facts, 
that it's capable of reasoning,
and if it's engaged in a description of the content of the video or causal reasoning and so on, 
it's not capable of reasoning. Give me a 1 for reasoning, a 0 for not reasoning
I will provide you with the system's output.
The output format is: Score: [01](regular expression)
The output format must strictly adhere to the following regular expression: "Score: 1"
'''

WWCH_OPENAI_API_KEY='sk-'
