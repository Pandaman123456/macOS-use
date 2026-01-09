import gradio as gr
from typing import Dict, List, Any

def create_agent_tab(app_instance) -> List[gr.components.Component]:
    """
    Creates the main chat interface for the agent.
    """
    with gr.Column(elem_classes="agent-container"):
        # Main Chat Area
        chatbot = gr.Chatbot(
            label="Agent",
            elem_id="chatbot",
            # bubble_full_width removed as it is deprecated/removed in Gradio 6.x
            height=500,
            type="messages"
        )

        # Input Area
        with gr.Row():
            task_input = gr.Textbox(
                show_label=False,
                placeholder="What would you like me to do on your Mac?",
                lines=2,
                scale=8,
                container=False,
                autofocus=True
            )
            run_button = gr.Button("Run", variant="primary", scale=1, size="lg")
            stop_button = gr.Button("Stop", variant="stop", scale=1, size="lg", interactive=False)

        # Settings & Advanced Options (Hidden by default)
        with gr.Accordion("Advanced Options & Examples", open=False):
            with gr.Row():
                refine_prompt_btn = gr.Button("Refine Prompt", size="sm")
                share_prompt = gr.Checkbox(
                    label="Share prompt anonymously",
                    value=app_instance.preferences.get("share_prompt", False)
                )
            
            with gr.Row():
                max_steps = gr.Slider(
                    minimum=1, maximum=100, value=25, step=1, label="Max Steps"
                )
                max_actions = gr.Slider(
                    minimum=1, maximum=20, value=5, step=1, label="Max Actions/Step"
                )

            # Examples
            gr.Markdown("### Examples")
            with gr.Row():
                quick_tasks = app_instance.example_categories.get("Quick Tasks", [])[:4]
                for example in quick_tasks:
                    gr.Button(example["name"], size="sm", variant="secondary").click(
                        fn=lambda p=example["prompt"]: p,
                        outputs=task_input
                    )

    # Custom CSS for the native app feel
    gr.HTML("""
        <style>
            .agent-container { padding: 0px; }
            #chatbot { border: none; background-color: transparent; }
            footer { display: none !important; }
            .gradio-container { max_width: 100% !important; margin: 0; padding: 0; }
        </style>
    """)

    return [
        task_input, refine_prompt_btn, share_prompt, max_steps, max_actions,
        run_button, stop_button, chatbot
    ]

def create_automations_tab(app_instance) -> List[gr.components.Component]:
    with gr.Row():
        with gr.Column(scale=2):
            automation_name = gr.Textbox(
                label="Automation Name",
                placeholder="Enter automation name"
            )
            automation_description = gr.Textbox(
                label="Description",
                placeholder="Enter automation description",
                lines=2
            )
            add_automation_btn = gr.Button("Add Automation", variant="primary")
            
            automation_list = gr.Dropdown(
                label="Select Automation",
                choices=list(app_instance.automations.keys()),
                interactive=True
            )
            
            agent_prompt = gr.Textbox(
                label="Agent Prompt",
                placeholder="Enter agent prompt",
                lines=3,
                interactive=True
            )
            
            with gr.Row():
                add_agent_btn = gr.Button("Add Agent", variant="primary")
                remove_agent_btn = gr.Button("Remove Selected Agent", variant="stop")
            
            run_automation_btn = gr.Button("Run Automation", variant="primary")
            
        with gr.Column(scale=3):
            agents_list = gr.List(
                label="Agents in Flow",
                headers=["#", "Prompt"],
                type="array",
                interactive=True,
                col_count=2
            )
            automation_output = gr.Textbox(
                label="Automation Output",
                lines=25,
                interactive=False,
                autoscroll=True
            )
    
    return [
        automation_name, automation_description, add_automation_btn,
        automation_list, agent_prompt, add_agent_btn, remove_agent_btn,
        run_automation_btn, agents_list, automation_output
    ]

def create_configuration_tab(app_instance) -> List[gr.components.Component]:
    # Get saved provider and model from preferences, or use defaults
    default_provider = app_instance.preferences.get("llm_provider", "OpenAI")
    
    llm_provider = gr.Dropdown(
        choices=list(app_instance.llm_models.keys()),
        label="LLM Provider",
        value=default_provider
    )
    
    # Get the models for the current provider
    available_models = app_instance.llm_models.get(default_provider, [])
    default_model = app_instance.preferences.get("llm_model", available_models[0] if available_models else None)
    
    llm_model = gr.Dropdown(
        choices=available_models,
        label="Model",
        value=default_model
    )
    
    api_key = gr.Textbox(
        label="API Key",
        type="password",
        placeholder="Enter your API key",
        value=app_instance.get_saved_api_key(default_provider)
    )
    
    # Add sharing preferences section
    gr.Markdown("### Sharing Settings")
    
    share_terminal = gr.Checkbox(
        label="Share terminal output anonymously",
        value=app_instance.preferences.get("share_terminal", True),
        info="Sharing terminal output helps us understand how the agent performs tasks."
    )
    
    return [llm_provider, llm_model, api_key, share_terminal]
