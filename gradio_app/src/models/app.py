import os
import json
import queue
import logging
import asyncio
import traceback
from pathlib import Path
from typing import Optional, Generator, AsyncGenerator, List
from dotenv import load_dotenv, set_key
import gradio as gr

from ..utils.logging_utils import setup_logging
from ..models.llm_models import LLM_MODELS, get_llm
from ..services.google_form import send_prompt_to_google_sheet
from ..config.example_prompts import EXAMPLE_CATEGORIES

# Import mlx_use from parent directory
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from mlx_use import Agent
from mlx_use.controller.service import Controller

class MacOSUseGradioApp:
    def __init__(self):
        self.agent = None
        self.controller = Controller()
        self.is_running = False
        self.log_queue = queue.Queue()
        self.setup_logging()
        self.terminal_buffer = []
        self.chat_history = []  # For storing chat messages (list of lists/tuples)
        self.message_queue = asyncio.Queue() # Queue for passing messages from callback to generator
        self.automations = {}
        self.preferences_file = Path(__file__).parent.parent.parent / 'preferences.json'
        self.preferences = self.load_preferences()
        self._cleanup_state()
        self.example_categories = EXAMPLE_CATEGORIES
        self.llm_models = LLM_MODELS
        self.current_task = None
        
        load_dotenv()

    def _cleanup_state(self):
        """Reset all state variables"""
        self.is_running = False
        self.agent = None
        self.current_task = None
        self.chat_history = []
        # Clear queues
        while not self.log_queue.empty():
            try: self.log_queue.get_nowait()
            except queue.Empty: break

        # We can't clear asyncio.Queue synchronously easily, but we can replace it
        self.message_queue = asyncio.Queue()

    def setup_logging(self):
        """Set up logging to capture terminal output"""
        setup_logging(self.log_queue)

    def get_terminal_output(self) -> str:
        """Get accumulated terminal output"""
        while True:
            try:
                log = self.log_queue.get_nowait()
                self.terminal_buffer.append(log)
            except queue.Empty:
                break
        return "".join(self.terminal_buffer)

    def stop_agent(self) -> tuple:
        """Stop the running agent"""
        if self.agent and self.is_running:
            self.is_running = False
            if hasattr(self.agent, '_stopped'):
                self.agent._stopped = True
            if self.current_task and not self.current_task.done():
                self.current_task.cancel()
            
            self._cleanup_state()
            
            # Add a system message indicating stop
            stop_msg = [None, "🛑 Agent stopped by user."]
            return (
                self.chat_history + [stop_msg],
                gr.update(interactive=True),
                gr.update(interactive=False)
            )
        return (
            [],
            gr.update(interactive=True),
            gr.update(interactive=False)
        )

    def add_automation(self, name: str, description: str) -> dict:
        if name in self.automations:
            raise ValueError(f"Automation '{name}' already exists")
        self.automations[name] = {"description": description, "agents": []}
        return gr.update(choices=list(self.automations.keys()))

    def add_agent_to_automation(self, automation_name: str, agent_prompt: str, position: int = -1) -> list:
        if automation_name not in self.automations:
            raise ValueError(f"Automation '{automation_name}' does not exist")
        new_agent = {"prompt": agent_prompt, "max_steps": 25, "max_actions": 1}
        if position == -1 or position >= len(self.automations[automation_name]["agents"]):
            self.automations[automation_name]["agents"].append(new_agent)
        else:
            self.automations[automation_name]["agents"].insert(position, new_agent)
        return self.automations[automation_name]["agents"]

    def remove_agent_from_automation(self, automation_name: str, agent_index: int) -> list:
        if automation_name not in self.automations:
            raise ValueError(f"Automation '{automation_name}' does not exist")
        if not isinstance(agent_index, int) or agent_index < 0 or agent_index >= len(self.automations[automation_name]["agents"]):
            raise ValueError(f"Invalid agent index {agent_index}")
        self.automations[automation_name]["agents"].pop(agent_index)
        return self.automations[automation_name]["agents"]

    def get_automation_agents(self, automation_name: str) -> list:
        if automation_name not in self.automations:
            raise ValueError(f"Automation '{automation_name}' does not exist")
        return self.automations[automation_name]["agents"]

    def save_api_key_to_env(self, provider: str, api_key: str) -> None:
        env_path = Path(__file__).parent.parent.parent.parent / '.env'
        if not env_path.exists(): env_path.touch()
        provider_to_env = {
            "OpenAI": "OPENAI_API_KEY", "Anthropic": "ANTHROPIC_API_KEY",
            "Google": "GEMINI_API_KEY", "alibaba": "DEEPSEEK_API_KEY"
        }
        env_var = provider_to_env.get(provider)
        if env_var and api_key: set_key(str(env_path), env_var, api_key)

    def get_saved_api_key(self, provider: str) -> str:
        provider_to_env = {
            "OpenAI": "OPENAI_API_KEY", "Anthropic": "ANTHROPIC_API_KEY",
            "Google": "GEMINI_API_KEY", "alibaba": "DEEPSEEK_API_KEY"
        }
        env_var = provider_to_env.get(provider)
        return os.getenv(env_var, "") if env_var else ""

    def load_preferences(self) -> dict:
        if self.preferences_file.exists():
            try:
                with open(self.preferences_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Failed to load preferences: {e}")
        return {"share_prompt": False, "share_terminal": True}

    def save_preferences(self) -> None:
        try:
            with open(self.preferences_file, 'w') as f:
                json.dump(self.preferences, f)
        except Exception as e:
            logging.error(f"Failed to save preferences: {e}")

    def update_share_prompt(self, value: bool) -> None:
        self.preferences["share_prompt"] = value
        self.save_preferences()

    def update_share_terminal(self, value: bool) -> None:
        self.preferences["share_terminal"] = value
        self.save_preferences()

    def update_llm_preferences(self, provider: str, model: str) -> None:
        self.preferences["llm_provider"] = provider
        self.preferences["llm_model"] = model
        self.save_preferences()

    async def run_automation(
        self,
        automation_name: str,
        llm_provider: str,
        llm_model: str,
        api_key: str,
    ) -> AsyncGenerator[tuple[str, dict, dict], None]:
        """Run an automation flow by executing its agents in sequence"""
        if automation_name not in self.automations:
            raise ValueError(f"Automation '{automation_name}' does not exist")

        # Save API key to .env file
        self.save_api_key_to_env(llm_provider, api_key)

        automation = self.automations[automation_name]
        self._cleanup_state()
        
        # Clear terminal buffer for new automation
        self.terminal_buffer = []
        
        try:
            for i, agent_config in enumerate(automation["agents"]):
                # Initialize LLM
                llm = get_llm(llm_provider, llm_model, api_key)
                if not llm:
                    raise ValueError(f"Failed to initialize {llm_provider} LLM")
                
                # Initialize agent
                self.agent = Agent(
                    task=agent_config["prompt"],
                    llm=llm,
                    controller=self.controller,
                    use_vision=False,
                    max_actions_per_step=agent_config["max_actions"]
                )
                
                self.is_running = True
                last_update = ""
                
                try:
                    # Start the agent run
                    agent_task = asyncio.create_task(self.agent.run(max_steps=agent_config["max_steps"]))
                    self.current_task = agent_task  # Store reference to current task
                    
                    # While the agent is running, yield updates periodically
                    while not agent_task.done() and self.is_running:
                        current_output = self.get_terminal_output()
                        if current_output != last_update:
                            # Check if we've had a "done" action
                            if "\"done\":" in current_output and "📄 Result:" in current_output:
                                logging.info("Detected 'done' action, stopping agent")
                                self.is_running = False
                                if not agent_task.done():
                                    agent_task.cancel()
                                break
                                
                            yield (
                                f"Running agent {i+1}/{len(automation['agents'])}\n{current_output}",
                                gr.update(interactive=False),
                                gr.update(interactive=True)
                            )
                            last_update = current_output
                        await asyncio.sleep(0.1)
                    
                    if not agent_task.done():
                        agent_task.cancel()
                        await asyncio.sleep(0.1)  # Allow time for cancellation
                    else:
                        result = await agent_task
                    
                    # Final update for this agent
                    final_output = self.get_terminal_output()
                    if final_output != last_update:
                        yield (
                            f"Completed agent {i+1}/{len(automation['agents'])}\n{final_output}",
                            gr.update(interactive=True) if i == len(automation["agents"]) - 1 else gr.update(interactive=False),
                            gr.update(interactive=False) if i == len(automation["agents"]) - 1 else gr.update(interactive=True)
                        )
                    
                except Exception as e:
                    error_details = f"Error Details:\n{traceback.format_exc()}"
                    self.terminal_buffer.append(f"\nError occurred in agent {i+1}:\n{str(e)}\n\n{error_details}")
                    yield (
                        "".join(self.terminal_buffer),
                        gr.update(interactive=True),
                        gr.update(interactive=False)
                    )
                    break
                
                self._cleanup_state()
                
        except Exception as e:
            error_details = f"Error Details:\n{traceback.format_exc()}"
            error_msg = f"Error occurred:\n{str(e)}\n\n{error_details}"
            yield (
                error_msg,
                gr.update(interactive=True),
                gr.update(interactive=False)
            )
            self._cleanup_state()

    async def get_llm_response(self, system_message: str, user_message: str, llm_provider: str, llm_model: str) -> str:
        try:
            api_key = self.get_saved_api_key(llm_provider)
            if not api_key: raise ValueError(f"No API key found for {llm_provider}")
            llm = get_llm(llm_provider, llm_model, api_key)
            if not llm: raise ValueError(f"Failed to initialize {llm_provider} LLM")
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [SystemMessage(content=system_message), HumanMessage(content=user_message)]
            response = await llm.ainvoke(messages)
            return response.content
        except Exception as e:
            logging.error(f"Error getting LLM response: {e}")
            return user_message

    def _step_callback(self, state: str, model_output: 'AgentOutput', step: int):
        """Callback to receive updates from the agent"""
        try:
            # Extract thoughts and actions
            thought = model_output.current_state.memory
            next_goal = model_output.current_state.next_goal
            actions = [action.model_dump_json(exclude_unset=True) for action in model_output.action]

            # Format the message
            content = f"**Step {step}**\n\n"
            content += f"💭 **Thought:** {thought}\n"
            content += f"🎯 **Goal:** {next_goal}\n\n"
            content += "**Actions:**\n"
            for action in actions:
                # Pretty print action JSON
                try:
                    action_dict = json.loads(action)
                    key = list(action_dict.keys())[0]
                    params = action_dict[key]
                    content += f"- **{key}**: `{params}`\n"
                except:
                    content += f"- `{action}`\n"

            # Format as a tuple for standard Gradio Chatbot (None for user, content for bot)
            message = [None, content]

            # Put into queue for the generator to pick up
            self.message_queue.put_nowait(message)
            
        except Exception as e:
            logging.error(f"Error in step callback: {e}")

    def _done_callback(self, history):
        """Callback when agent is done"""
        try:
            final_result = history.final_result()
            if final_result:
                msg = [None, f"✅ **Task Completed!**\n\n{final_result}"]
            else:
                msg = [None, "✅ **Task Completed!**"]
            self.message_queue.put_nowait(msg)
            # Signal end of stream
            self.message_queue.put_nowait(None)
        except Exception as e:
            logging.error(f"Error in done callback: {e}")
            self.message_queue.put_nowait(None)

    async def run_agent(
        self,
        task: str,
        max_steps: int,
        max_actions: int,
        llm_provider: str,
        llm_model: str,
        api_key: str,
        share_prompt: bool,
        share_terminal: bool
    ) -> AsyncGenerator[tuple[list, dict, dict], None]:
        """Run the agent and yield chat history updates"""
        self._cleanup_state()
        
        # Initial User Message: [task, None] means user said 'task', bot said nothing yet
        user_msg = [task, None]
        self.chat_history.append(user_msg)
        yield (
            self.chat_history,
            gr.update(interactive=False),
            gr.update(interactive=True)
        )

        try:
            if not task.strip():
                # Append error message as bot response
                self.chat_history.append([None, "⚠️ Please enter a task description."])
                yield self.chat_history, gr.update(interactive=True), gr.update(interactive=False)
                return

            self.save_api_key_to_env(llm_provider, api_key)
            llm = get_llm(llm_provider, llm_model, api_key)

            if not llm:
                self.chat_history.append([None, f"❌ Failed to initialize {llm_provider} LLM."])
                yield self.chat_history, gr.update(interactive=True), gr.update(interactive=False)
                return

            # Initialize Agent with callbacks
            self.agent = Agent(
                task=task,
                llm=llm,
                controller=self.controller,
                use_vision=False,
                max_actions_per_step=max_actions,
                register_new_step_callback=self._step_callback,
                register_done_callback=self._done_callback
            )
            
            self.is_running = True
            
            # Start agent in background task
            agent_task = asyncio.create_task(self.agent.run(max_steps=max_steps))
            self.current_task = agent_task

            # Poll the message queue
            while self.is_running:
                try:
                    # Wait for new messages with a timeout to allow checking task status
                    message = await asyncio.wait_for(self.message_queue.get(), timeout=0.1)

                    if message is None: # Signal to stop
                        break

                    self.chat_history.append(message)
                    yield (
                        self.chat_history,
                        gr.update(interactive=False),
                        gr.update(interactive=True)
                    )
                except asyncio.TimeoutError:
                    if agent_task.done():
                        break
                    continue
            
            # Check for errors if task finished unexpectedly
            if agent_task.done() and not agent_task.cancelled():
                try:
                    await agent_task
                except Exception as e:
                    error_msg = [None, f"❌ **Error:** {str(e)}"]
                    self.chat_history.append(error_msg)
                    yield self.chat_history, gr.update(interactive=True), gr.update(interactive=False)

        except Exception as e:
            error_msg = [None, f"❌ **System Error:** {str(e)}"]
            self.chat_history.append(error_msg)
            yield self.chat_history, gr.update(interactive=True), gr.update(interactive=False)
        finally:
            self._cleanup_state()
            yield self.chat_history, gr.update(interactive=True), gr.update(interactive=False)
