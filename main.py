import os
os.environ["OPENAI_API_KEY"] = ""
from flaml.autogen.agentchat.assistant_agent import AssistantAgent
from miniwob_agent import MiniWobUserProxyAgent
from flaml.autogen import oai
import argparse
import flaml

parser = argparse.ArgumentParser(description="input task")
parser.add_argument(
    "--problem", type=str, default="click-button-sequence", help="task"
)
args = parser.parse_args()
problem = args.problem

Configlist=[{"model":"gpt-3.5-turbo-16k", "api_key":""}]


task_list = [
    "choose-list",
    "click-button-sequence",
    "click-button",
    "click-checkboxes-large",
    "click-checkboxes-soft",
    "click-checkboxes-transfer",
    "click-checkboxes",
    "click-collapsible-2",
    "click-collapsible",
    "click-color",
    "click-dialog-2",
    "click-dialog",
    "click-link",
    "click-menu",
    "click-option",
    "click-scroll-list",
    "click-shades",
    "click-shape",
    "click-tab-2-hard",
    "click-tab-2",
    "click-tab",
    "click-test-2",
    "click-test",
    "click-widget",
    "count-shape",
    "email-inbox-forward-nl-turk",
    "email-inbox-forward-nl",
    "email-inbox-nl-turk",
    "email-inbox",
    "enter-date",
    "enter-password",
    "enter-text-dynamic",
    "enter-text",
    "enter-time",
    "focus-text-2",
    "focus-text",
    "grid-coordinate",
    "login-user-popup",
    "login-user",
    "navigate-tree",
    "search-engine",
    "simple-algebra",
    "social-media-all",
    "social-media-some",
    "social-media",
    "terminal",
    "use-spinner",
]

for task in task_list:
    for _ in range(10):

        assistant = AssistantAgent(
            name="assistant", 
            system_message="You are an autoregressive language model that completes user's sentences. You should not conversate with user.",
            llm_config={
                "request_timeout": 600,
                "seed": 42,
                "config_list": Configlist,
            }
        )

        MiniWob = MiniWobUserProxyAgent(
            name="MiniWobUserProxyAgent", 
            human_input_mode="NEVER",
            problem = task,
            headless=False,
            env_name= task,
            rci_plan_loop=0,
            rci_limit=1,
            llm="chatgpt",
            state_grounding=False,
        )

        assistant.reset()

        MiniWob.initiate_chat(assistant)
