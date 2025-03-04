"""
Prompt Loader Module
"""

import os
import yaml


class PromptLoader:
    """Prompt Loader Class"""

    def __init__(self, templates_dir: str):
        """Initialize the prompt loader

        Args:
            templates_dir: Path to the template directory
        """
        self.templates_dir = templates_dir

    def load_prompt(self, prompt_key: str) -> str:
        """Load the specified prompt template

        Args:
            prompt_key: Key (filename without extension) of the prompt template

        Returns:
            Formatted prompt template string
        """
        template_path = os.path.join(self.templates_dir, f"{prompt_key}.yaml")

        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Prompt template not found: {template_path}")

        with open(template_path, 'r', encoding='utf-8') as f:
            template_data = yaml.safe_load(f)

        return template_data['prompt']

    def format_prompt(self, template: str, polynomial: str) -> str:
        """Format prompt template

        Args:
            template: Prompt template string
            polynomial: Polynomial expression to insert

        Returns:
            Formatted prompt string
        """
        return template.replace("<INPUT>", polynomial)
