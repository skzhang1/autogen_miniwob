import os
os.environ["OPENAI_API_KEY"] = ""
from flaml.autogen.agent.assistant_agent import AssistantAgent
from miniwob_agent import MiniWobUserProxyAgent
from flaml import oai
import argparse


parser = argparse.ArgumentParser(description="input task")
parser.add_argument(
    "--problem", type=str, default="click-button-sequence", help="task"
)
args = parser.parse_args()
problem = args.problem

config_list = oai.config_list_gpt4_gpt35()
config_list[0]['model'] = "gpt-3.5-turbo" 
# print(config_list)

for _ in range(10):

    assistant = AssistantAgent(
        name="assistant", 
        system_message="You are a helpful assistant.",
        oai_config={
            "request_timeout": 600,
            "seed": 42,
            "config_list": config_list,
        }
    )

    MiniWob = MiniWobUserProxyAgent(
        name="MiniWobUserProxyAgent", 
        # human_input_mode="NEVER",
        # use_docker=False,
        problem = problem,
        headless=False,
    )

    assistant.reset()

    MiniWob.initiate_chat(assistant)
