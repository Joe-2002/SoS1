"""
Experiment Controller Module

This module acts as the main controller, responsible for coordinating and managing the entire experiment process, including:
1. Model prediction execution
2. Results analysis
3. Experiment configuration management
4. Logging
"""

from analyze_results import analyze_results
from sos_api_caller import SOSPredictor
import os
import sys
import json
import logging
from datetime import datetime
from typing import Optional
import pandas as pd
from prompt_loader import PromptLoader

# Set import path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


class ExperimentController:
    """Experiment Controller class, responsible for coordinating the entire experiment process"""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the experiment controller"""
        self.base_path = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        self.setup_logging()
        self.config = self.load_config(config_path)
        self.predictor = None
        self.current_experiment_id = None
        self.prompt_loader = PromptLoader(
            os.path.join(self.base_path, 'data', 'prompts'))

    def setup_logging(self):
        """Set up the logging system"""
        log_dir = os.path.join(self.base_path, 'Logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(
            log_dir, f'experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration file"""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.logger.info(f"User configuration loaded: {config_path}")
                return config
            except Exception as e:
                self.logger.error(f"Failed to load configuration file: {str(e)}")
        return {}

    def initialize_predictor(self):
        """Initialize the predictor"""
        try:
            self.predictor = SOSPredictor()
            # Pass experiment configuration to the predictor
            self.predictor.experiment_config = self.config
            self.logger.info("Predictor initialized successfully")
        except Exception as e:
            self.logger.error(f"Predictor initialization failed: {str(e)}")
            raise

    def run_experiment(self):
        """Run the main experiment process"""
        try:
            self.logger.info("Starting new experiment")
            self.current_experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Ensure input file exists
            input_file = self.config['input_file']
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Input file not found: {input_file}")

            # Load prompt configurations
            prompt_keys = self.config['prompt_keys']
            prompt_templates = []
            for key in prompt_keys:
                template = self.prompt_loader.load_prompt(key)
                if template:
                    prompt_templates.append(template)
                else:
                    self.logger.warning(f"Could not load prompt template: {key}")

            if not prompt_templates:
                raise ValueError("No prompt templates were successfully loaded")

            self.logger.info(f"Prompt templates loaded: {', '.join(prompt_keys)}")

            # Initialize predictor
            if not self.predictor:
                self.initialize_predictor()

            # Read data
            df = pd.read_json(input_file, lines=True)
            if df.empty:
                raise ValueError("Input file is empty")

            # If sample size limit is set, take the first N data points
            sample_size = self.config.get('sample_size')
            if sample_size is not None and sample_size < len(df):
                self.logger.info(f"Using the first {sample_size} data points")
                df = df.head(sample_size)

            # Execute predictions
            self.logger.info("Starting model prediction phase")
            results_df = None

            try:
                results_df = self.predictor.process_file(
                    df,
                    self.config['model_configs'],
                    prompt_templates,
                    self.config['rounds']
                )

                if results_df is not None:
                    # Run analysis
                    self.logger.info("Starting results analysis phase")
                    output_dir = os.path.join(
                        self.base_path,
                        "Analysis",
                        f"{self.config['analysis_dir_prefix']}{self.current_experiment_id}"
                    )

                    try:
                        analyze_results(results_df, output_dir)
                        self.logger.info(f"Analysis complete, results saved to: {output_dir}")
                    except Exception as e:
                        self.logger.error(f"Error during result analysis: {str(e)}")
                else:
                    self.logger.error("Model prediction did not return valid results")

            except Exception as e:
                self.logger.error(f"Model processing failed: {str(e)}")
                return

            self.logger.info("Experiment completed")

        except Exception as e:
            self.logger.error(f"Experiment execution failed: {str(e)}")
            raise

    def cleanup(self):
        """Clean up resources"""
        self.predictor = None
        self.current_experiment_id = None


def main():
    """Main function"""
    # Quick configuration area
    config = {
        # Data configuration
        'input_file': os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data',
            'evaluation.jsonl'
        ),
        'sample_size': None,  # Set the number of samples to process, None means use all data

        # Model configuration
        'model_configs': [
            # DeepSeek Models (Bailian API call)
            {'name': 'deepseek-r1-distill-qwen-1.5b', 'temperature': None,
                'try_channels': ['bailian']},
            # {'name': 'deepseek-r1-distill-qwen-7b', 'temperature': None,
            #     'try_channels': ['bailian']},
            # {'name': 'deepseek-r1-distill-qwen-14b', 'temperature': None,
            #     'try_channels': ['bailian']},
            # {'name': 'deepseek-r1-distill-qwen-32b', 'temperature': None,
            #     'try_channels': ['bailian']},
            # {'name': 'deepseek-r1-distill-llama-70b', 'temperature': None,
            #     'try_channels': ['bailian']},
            {'name': 'deepseek-r1-distill-llama-8b', 'temperature': None,
                'try_channels': ['bailian']},
            # {'name': 'deepseek-r1', 'temperature': None,
            #     'try_channels': ['bailian']},
        ],

        # Prompt configuration
        'prompt_keys': ['SoS_Plain', 'SoS_Simple', 'SoS_Reasoning'],  # Use prompt templates from data/prompts directory
        # Experiment configuration
        'rounds': 1,  # Number of test rounds per sample
        'analysis_dir_prefix': 'Analysis_'  # Prefix for analysis results directory
    }

    # Create controller instance
    controller = ExperimentController()
    controller.config = config  # Use quick configuration

    try:
        # Run experiment
        controller.run_experiment()
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        controller.logger.warning("Experiment interrupted by user")
    except Exception as e:
        print(f"\nExperiment execution error: {str(e)}")
        controller.logger.error(f"Experiment execution error: {str(e)}")
    finally:
        # Clean up resources
        controller.cleanup()


if __name__ == "__main__":
    main()
