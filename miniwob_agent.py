import json
from prompt import Prompt
import time
import openai
from pathlib import Path
from selenium.webdriver.common.keys import Keys
import os
import logging
import random
import computergym
import gym
from computergym.miniwob.miniwob_interface.action import (
    MiniWoBType,
    MiniWoBElementClickId,
    MiniWoBElementClickXpath,
    MiniWoBElementClickOption,
    MiniWoBMoveXpath,
)
import re
from flaml.autogen.agentchat import ResponsiveAgent
from flaml.autogen.agentchat.agent import Agent
from typing import Any, Callable, Dict, List, Optional, Union

class MiniWobUserProxyAgent(ResponsiveAgent):
    def __init__(
        self,
        # from main
        env_name: str,
        headless_miniwob = True,
        # origin
        env = None,
        rci_plan_loop: int = 1,
        rci_limit: int = 1,
        llm="chatgpt",
        with_task=True,
        state_grounding=True,
        # autogen
        name= "MinWobAgent",
        is_termination_msg = lambda x: "terminate" in x.get("content").lower(),  
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "NEVER",
        function_map: Optional[Dict[str, Callable]] = None,
        code_execution_config: Optional[Union[Dict, bool]] = None,
        oai_config: Optional[Union[Dict, bool]] = False,
        system_message: Optional[str] = "",
        problem=None,
        headless=False,
        **kwargs,
    ) -> None:
        # autogen

        super().__init__(
            name=name,
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply = max_consecutive_auto_reply,
            human_input_mode = human_input_mode,
            function_map = function_map,
            code_execution_config = code_execution_config,
            llm_config = oai_config,
            system_message = system_message,
            **kwargs,
        )
        
        self.register_auto_reply(trigger="miniwob_assistant", reply_func = MiniWobUserProxyAgent.reply_miniwob, position = 1)        
        with open("config.json") as config_file:
            api_key = json.load(config_file)["api_key"]
            openai.api_key = api_key
        
        # main
        self.env_name = env_name
        self.real_env = gym.make("MiniWoBEnv-v0", env_name=env_name, headless=headless_miniwob)
        self.recipient = None
        self.silent = False
        
        # rci
        self.rci_limit = rci_limit
        self.rci_plan_loop = rci_plan_loop
        self.llm = llm
        self.prompt = Prompt(env=env_name)
        self.state_grounding = state_grounding

        self.load_model()

        self.html_state = ""
        self.task = ""
        self.with_task = with_task
        self.current_plan = ""
        self.past_plan = []
        self.past_instruction = []
        self.custom_gaol = False

        self.history_name = time.strftime("%Y%m%d-%H%M%S")
        config_string = (
            f"erci{rci_plan_loop}_state{self.state_grounding}_irci{rci_limit}"
        )
        if self.prompt.example_prompt:
            self.file_path = Path(
                f"history/{self.llm}/{env_name}/{config_string}/few-shot/{self.history_name}.txt"
            )
        else:
            self.file_path = Path(
                f"history/{self.llm}/{env_name}/{config_string}/zero-shot/{self.history_name}.txt"
            )
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # flow_control
        self.plan_stage = True
        self.criticizm = True
        self.unexecuted_steps= 0
        self.ask_action = True
        self.judge_action = True

    def convert_to_miniwob_action(self, instruction: str):
        instruction = instruction.split(" ")
        inst_type = instruction[0]
        inst_type = inst_type.lower()

        if inst_type == "type":
            characters = " ".join(instruction[1:])
            characters = characters.replace('"', "")
            return MiniWoBType(characters)
        elif inst_type == "clickid":
            element_id = " ".join(instruction[1:])
            return MiniWoBElementClickId(element_id)
        elif inst_type == "press":
            key_type = instruction[1].lower()
            if key_type == "enter":
                return MiniWoBType("\n")
            elif key_type == "space":
                return MiniWoBType(" ")
            elif key_type == "arrowleft":
                return MiniWoBType(Keys.LEFT)
            elif key_type == "arrowright":
                return MiniWoBType(Keys.RIGHT)
            elif key_type == "backspace":
                return MiniWoBType(Keys.BACKSPACE)
            elif key_type == "arrowup":
                return MiniWoBType(Keys.UP)
            elif key_type == "arrowdown":
                return MiniWoBType(Keys.DOWN)
            else:
                raise NotImplemented
        elif inst_type == "movemouse":
            xpath = " ".join(instruction[1:])
            return MiniWoBMoveXpath(xpath)
        elif inst_type == "clickxpath":
            xpath = " ".join(instruction[1:])
            return MiniWoBElementClickXpath(xpath)
        elif inst_type == "clickoption":
            xpath = " ".join(instruction[1:])
            return MiniWoBElementClickOption(xpath)
        else:
            raise ValueError("Invalid instruction")
        
    def load_model(self):
        with open("config.json") as config_file:
            api_key = json.load(config_file)["api_key"]
            openai.api_key = api_key
        if self.llm == "chatgpt":
            self.model = "gpt-3.5-turbo"
        elif self.llm == "gpt4":
            self.model = "gpt-4"
        elif self.llm == "davinci":
            self.model = "text-davinci-003"
        elif self.llm == "ada":
            self.model = "ada"
        elif self.llm == "babbage":
            self.model = "babbage"
        elif self.llm == "curie":
            self.model = "curie"
        elif self.llm == "davinci1":
            self.model = "davinci"
        elif self.llm == "davinci2":
            self.model = "text-davinci-002"
        else:
            raise NotImplemented

    def update_html_state(self, state: str):
        self.html_state = state

        return

    def set_goal(self, goal: str):
        self.custom_gaol = True
        self.task = goal

        return
   
    def check_regex(self, instruciton):
        return (
            (not re.search(self.prompt.clickxpath_regex, instruciton, flags=re.I))
            and (not re.search(self.prompt.chatgpt_type_regex, instruciton, flags=re.I))
            and (not re.search(self.prompt.davinci_type_regex, instruciton, flags=re.I))
            and (not re.search(self.prompt.press_regex, instruciton, flags=re.I))
            and (not re.search(self.prompt.clickoption_regex, instruciton, flags=re.I))
            and (not re.search(self.prompt.movemouse_regex, instruciton, flags=re.I))
        )

    def process_instruction(self, instruciton: str):
        end_idx = instruciton.find("`")
        if end_idx != -1:
            instruciton = instruciton[:end_idx]

        instruciton = instruciton.replace("`", "")
        instruciton = instruciton.replace("\n", "")
        instruciton = instruciton.replace("\\n", "\n")
        instruciton = instruciton.strip()
        instruciton = instruciton.strip("'")

        return instruciton

    def get_plan_step(self):
        idx = 1
        while True:
            if (str(idx) + ".") not in self.current_plan:
                return (idx - 1) + 1
            idx += 1
                   
    def get_html_state(self, env_name, states):
        extra_html_task = [
            "click-dialog",
            "click-dialog-2",
            "use-autocomplete",
            "choose-date",
        ]

        html_body = states[0].html_body
        if env_name in extra_html_task:
            html_body += states[0].html_extra
        return html_body

    def update_action(self, action = None):
        if self.prompt.update_action and self.state_grounding:
            pt = self.prompt.update_action
            message = self.get_response(pt)
            action = message

        return action

    def current_plan_prompt(self):
        pt = "\n\n"
        pt += "Here is a plan you are following now.\n"
        pt += f"{self.current_plan}"
        pt += "\n\n"

        return pt

    def instruction_history_prompt(self):
        pt = "\n\n"
        pt += "We have a history of instructions that have been already executed by the autonomous agent so far.\n"
        if not self.past_instruction:
            pt += "No instruction has been executed yet."
        else:
            for idx, inst in enumerate(self.past_instruction):
                pt += f"{idx+1}: "
                pt += inst
                pt += "\n"
        pt += "\n\n"

        return pt

    def webpage_state_prompt(self, init_plan: bool = False, with_task=False):
        pt = "\n\n"
        pt += "Below is the HTML code of the webpage where the agent should solve a task.\n"
        pt += self.html_state
        pt += "\n\n"
        if self.prompt.example_prompt and (init_plan or self.rci_plan_loop == -1):
            pt += self.prompt.example_prompt
            pt += "\n\n"
        if with_task:
            pt += "Current task: "
            pt += self.task
            pt += "\n"

        return pt

    def save_result(self, value):
        path_dir = os.path.join("./result", self.env_name+".json")
        if os.path.exists(path_dir):
            with open(path_dir, 'r') as f:
                data = json.load(f)
        else:
            data = {}

        if 'value' in data:
            if value >0:
                data['value'] += 1
        else:
            data['value'] = 0
        print(self.env_name)
        print("success rate", data['value'])
        with open(path_dir, 'w') as f:
            json.dump(data, f)
            
    def rci_plan(self, pt=None):
        # pt += "\n\nFind problems with this plan for the given task compared to the example plans.\n\n"
        pt = "\n\nFind problems with this plan for the given task compared to the example plans.\n\n"
        criticizm = self.get_response(pt)
        pt += criticizm

        # pt += "\n\nBased on this, what is the plan for the agent to complete the task?\n\n"
        pt = "\n\nBased on this, what is the plan for the agent to complete the task?\n\n"
        # pt += self.webpage_state_prompt()
        plan = self.get_response(pt)

        return pt, plan

    def rci_action(self, instruciton):
        instruciton = self.process_instruction(instruciton)

        loop_num = 0
        while self.check_regex(instruciton):
            if loop_num >= self.rci_limit:
                raise ValueError("Action RCI failed")

            pt = self.prompt.rci_action_prompt
            instruciton = self.get_response(pt)

            # pt += instruciton
            instruciton = self.process_instruction(instruciton)

            loop_num += 1

        return instruciton

    def initialize_plan(self):
        if not self.custom_gaol:
            if self.with_task:
                self.initialize_task()

        if not self.prompt.init_plan_prompt or self.rci_plan_loop == -1:
            return

        pt = self.prompt.base_prompt
        pt += self.webpage_state_prompt(True, with_task=self.with_task)
        pt += self.prompt.init_plan_prompt

        # in receive
        message = "\n" + self.get_response(pt)   
        pt += message

        for _ in range(self.rci_plan_loop):
            pt, message = self.rci_plan(pt)
            pt += message

        self.current_plan = message
        return

    def generate_action(self) -> str:
        # pt = self.prompt.base_prompt # check
        pt = self.webpage_state_prompt(with_task=self.with_task)
        if self.prompt.init_plan_prompt and self.rci_plan_loop != -1:
            pt += self.current_plan_prompt()
        pt += self.instruction_history_prompt()
        if self.past_instruction:
            update_action_prompt = self.prompt.action_prompt.replace(
                "{prev_inst}", self.past_instruction[-1]
            )
            if len(self.past_instruction) == 1:
                update_action_prompt = self.prompt.action_prompt.replace(
                    "{order}", "2nd"
                )
            elif len(self.past_instruction) == 2:
                update_action_prompt = self.prompt.action_prompt.replace(
                    "{order}", "3rd"
                )
            else:
                update_action_prompt = self.prompt.action_prompt.replace(
                    "{order}", f"{len(self.past_instruction)+1}th"
                )

            action_prompt = update_action_prompt
        else:
            action_prompt = self.prompt.first_action_prompt

        if self.rci_plan_loop == -1:
            action_prompt = "Based on the task, " + action_prompt
        else:
            action_prompt = (
                "Based on the plan and the history of instructions executed so far, "
                + action_prompt
            )

        pt += action_prompt
        message = self.get_response(pt)
        
        action = self.process_instruction(message)
        action = self.update_action(action)
        # pt, instruction = self.rci_action(pt=pt, instruciton=message)
        action = self.rci_action(action)

        self.past_instruction.append(action)

        return action

    def receive(
        self,
        message: Union[Dict, str],
        sender: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        
        self._process_received_message(message, sender, silent)
        if request_reply is False or request_reply is None and self.reply_at_receive[sender] is False:
            return
        reply = self.generate_reply(messages=self.chat_messages[sender], sender=sender)
        if reply is not None:
            self.send(reply, sender, silent=silent)
            
    def initiate_chat(
        self,
        recipient: "ResponsiveAgent",
        clear_history: Optional[bool] = False,
        silent: Optional[bool] = False,
        **context,
    ):
        # main
        self._prepare_chat(recipient, clear_history)
        states = self.real_env.reset(seeds=[random.random()], record_screenshots=True)
        self.set_goal(states[0].utterance)
        html_state = self.get_html_state(self.env_name, states)
        self.update_html_state(html_state)
        if not self.custom_gaol:
            if self.with_task:
                self.initialize_task()

        if not self.prompt.init_plan_prompt or self.rci_plan_loop == -1:
            return

        pt = self.prompt.base_prompt
        pt += self.webpage_state_prompt(True, with_task=self.with_task)
        pt += self.prompt.init_plan_prompt
        self.send(pt, recipient, silent=silent)
                
    def reply_miniwob(self, messages: List[Dict], sender: Optional[Agent] = None, config: Optional[Any] = None) -> Union[str, Dict]:
        messages = messages[-1]
        if not isinstance(messages,str): 
            messages = messages.get("content", "")
            
        if self.plan_stage:         
            ### rci
            if self.rci_plan_loop!=0:
                if self.criticizm:
                    reply = "\n\nFind problems with this plan for the given task compared to the example plans.\n\n"
                    self.criticizm = False
                    return True, reply
                else:
                    reply = "\n\nBased on this, what is the plan for the agent to complete the task?\n\n"
                    self.criticizm = True
                    self.rci_plan_loop -=1
                    return True, reply

            messages = "\n" + messages
            self.current_plan = messages
            self.plan_stage = False
        
        ############################ action stage
        if not self.plan_stage:
            if self.ask_action:
                reply = self.webpage_state_prompt(with_task=self.with_task)
                if self.prompt.init_plan_prompt and self.rci_plan_loop != -1:
                    reply += self.current_plan_prompt()
                reply += self.instruction_history_prompt()
                if self.past_instruction:
                    update_action_prompt = self.prompt.action_prompt.replace(
                        "{prev_inst}", self.past_instruction[-1]
                    )
                    if len(self.past_instruction) == 1:
                        update_action_prompt = self.prompt.action_prompt.replace(
                            "{order}", "2nd"
                        )
                    elif len(self.past_instruction) == 2:
                        update_action_prompt = self.prompt.action_prompt.replace(
                            "{order}", "3rd"
                        )
                    else:
                        update_action_prompt = self.prompt.action_prompt.replace(
                            "{order}", f"{len(self.past_instruction)+1}th"
                        )

                    action_prompt = update_action_prompt
                else:
                    action_prompt = self.prompt.first_action_prompt

                if self.rci_plan_loop == -1:
                    action_prompt = "Based on the task, " + action_prompt
                else:
                    action_prompt = (
                        "Based on the plan and the history of instructions executed so far, "
                        + action_prompt
                    )
                reply += action_prompt
                self.ask_action = False
                return True, reply

            # update action
            if self.judge_action and self.prompt.update_action and self.state_grounding:
                reply = self.prompt.update_action
                self.judge_action = False
                return True, reply
        
            # rci
            if self.rci_limit!=0:
                instruciton = self.process_instruction(messages)
                if self.check_regex(instruciton):      
                    reply = self.prompt.rci_action_prompt
                    self.rci_limit -=1
                    return True, reply
                else:
                    instruciton = messages
                    
            # execute
            instruction = self.process_instruction(messages)
            self.past_instruction.append(instruction)
            try:
                miniwob_action = self.convert_to_miniwob_action(instruction)

                states, rewards, dones, _ = self.real_env.step([miniwob_action])
            except (ValueError, TypeError):
                print("Invalid action or rci action fail")
                rewards = [0]
                dones = [True]
                
            if rewards[0] !=0 or all(dones):   
                if rewards[0] > 0:
                    self.save_result(1)
                    print("SUCCESS!!!!")
                    self.real_env.close()
                    return True, None
                else:
                    self.save_result(-1)
                    print("Fail!!!!")     
                    self.real_env.close()   
                    return True, None
            else:
                html_state = self.get_html_state(self.env_name, states)
                self.update_html_state(html_state)
                self.ask_action = True
                return False, None