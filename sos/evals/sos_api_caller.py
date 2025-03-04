from typing import Dict, List, Union, Optional, Tuple
import pandas as pd
import openai
import httpx
import json
import urllib3
import ssl
import os
from datetime import datetime
import re
import matplotlib.pyplot as plt
import seaborn as sns
from plot_style import setup_plot_style
import logging
import socket
import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from queue import Queue, Empty
import sys
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager, Pool
from concurrent.futures import ProcessPoolExecutor

# Set global plot style
setup_plot_style()


class ProgressBar:
    def __init__(self, total: int, prefix: str = '', length: int = 50):
        """Initialize progress bar

        Args:
            total: Total number of items
            prefix: Prefix string
            length: Bar length
        """
        self.total = total
        self.prefix = prefix
        self.length = length
        self.current = 0
        self.correct = 0
        self.lock = Lock()

    def update(self, current: int, correct: int = None):
        """Update progress bar

        Args:
            current: Current progress
            correct: Number of correct predictions
        """
        with self.lock:
            self.current = current
            if correct is not None:
                self.correct = correct
            percentage = float(current) / float(self.total)
            filled_length = int(self.length * percentage)
            bar = '=' * filled_length + '-' * (self.length - filled_length)
            accuracy = self.correct / current if current > 0 else 0

            # Clear current line and update progress
            sys.stdout.write('\033[K')  # Clear current line
            sys.stdout.write(
                f'\r{self.prefix} [{bar}] {current}/{self.total} ({accuracy:.1%})')
            sys.stdout.flush()


class ModelProgressTracker:
    """Tracks progress and statistics for a specific model"""

    def __init__(self, model_name: str, total_tasks: int):
        """Initialize the tracker

        Args:
            model_name: Name of the model being tracked
            total_tasks: Total number of tasks to process
        """
        self.model_name = model_name
        self.total_tasks = total_tasks
        self.manager = Manager()
        self.completed_tasks = self.manager.Value('i', 0)
        self.correct_predictions = self.manager.Value('i', 0)
        self.total_processed = self.manager.Value('i', 0)
        self.response_times = self.manager.list()
        self.progress_lock = self.manager.Lock()
        self.current_sample_count = self.manager.Value('i', 0)  # New: current sample count

    def update(self, is_correct: Optional[bool] = None, output: Optional[dict] = None) -> Tuple[int, float]:
        """Update progress with a new task result

        Args:
            is_correct: Whether the prediction was correct
            output: Output data from the model

        Returns:
            Tuple of (processed_tasks, accuracy)
        """
        with self.progress_lock:
            self.completed_tasks.value += 1
            self.current_sample_count.value += 1  # Update current sample count

            if is_correct is not None:
                if is_correct:
                    self.correct_predictions.value += 1
                self.total_processed.value += 1

            if output and 'api_time' in output:
                self.response_times.append(output['api_time'])

            # Update progress display
            self._display_progress(output, is_correct)

            accuracy = self.correct_predictions.value / \
                self.total_processed.value if self.total_processed.value > 0 else 0
            return self.completed_tasks.value, accuracy

    def _display_progress(self, output: Optional[dict] = None, is_correct: Optional[bool] = None):
        """Display progress information

        Args:
            output: Output data from the model
            is_correct: Whether the prediction was correct
        """
        percentage = float(self.completed_tasks.value) / \
            float(self.total_tasks)
        bar_length = 50
        filled_length = int(bar_length * percentage)
        bar = '=' * filled_length + '-' * (bar_length - filled_length)

        sys.stdout.write('\033[K')  # Clear current line
        sys.stdout.write(
            f'\r{self.model_name:<50} [{bar}] {self.completed_tasks.value}/{self.total_tasks}')

        if self.total_processed.value > 0:
            accuracy = self.correct_predictions.value / self.total_processed.value
            sys.stdout.write(f' ({accuracy:.1%})')

        sys.stdout.flush()

        if output:
            print()  # Line break
            if output['result'] is None:
                print("\033[91m[Error/Invalid Output]\033[0m")

            # Optimize input display
            input_poly = output.get('input', '')
            if input_poly:
                # Only show first 10 characters, if longer show ellipsis
                if len(input_poly) > 10:
                    input_poly = input_poly[:10] + "..."
                print(f"Input: {input_poly}")

            response = output.get('response', '').strip()
            if response:
                response_lines = response.split('\n')
                conclusion_lines = []
                for line in response_lines[-3:]:  # Only look at last 3 lines
                    if any(key in line.lower() for key in ['conclusion', 'therefore', 'boxed', 'final']):
                        conclusion_lines.append(line)

                if conclusion_lines:
                    print("Conclusion:", conclusion_lines[-1])
                else:
                    # If no conclusion found, only show first 50 characters of last line
                    last_line = response_lines[-1][:50] + "..." if len(
                        response_lines[-1]) > 50 else response_lines[-1]
                    print("Response:", last_line)

                result_mark = '\033[92m✓\033[0m' if is_correct else '\033[91m✗\033[0m'
                print(
                    f"Result: {output.get('result')} | Actual: {output.get('actual')} [{result_mark}]")
            if output.get('api_time', 0) > 0:
                print(f"Time: {output['api_time']:.1f}s")

            # Display cumulative statistics
            print(f"Total samples: {self.current_sample_count.value}")
            if self.total_processed.value > 0:
                print(
                    f"Cumulative accuracy: {self.correct_predictions.value/self.total_processed.value:.1%}")
                print("-" * 50)
            print()  # Leave space for next progress bar

    def get_final_stats(self) -> dict:
        """Get final statistics"""
        try:
            # Calculate average response time, excluding None values
            valid_times = [t for t in self.response_times if t is not None]
            avg_time = sum(valid_times) / \
                len(valid_times) if valid_times else 0

            # Ensure no division by zero
            total_processed = max(1, self.total_processed.value)

            return {
                'total': self.current_sample_count.value,  # Use actual processed sample count
                'correct': self.correct_predictions.value,
                'accuracy': self.correct_predictions.value / total_processed,
                'avg_time': avg_time
            }
        except Exception as e:
            print(f"Error calculating statistics: {str(e)}")
            return {
                'total': self.total_tasks,
                'correct': 0,
                'accuracy': 0.0,
                'avg_time': 0.0
            }


class APIConfig:
    """API Configuration Class"""

    def __init__(self):
        # Disable SSL verification warnings
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # API key configuration
        self.deepseek_key = ""
        self.siliconflow_key = ""
        self.qwen_key = ""
        self.chatgpt_key = ""
        self.volc_key = ""  # Volcano Engine API key

        # Experiment configuration
        self.experiment_config = None

        # API base URLs
        self.deepseek_base = "https://api.deepseek.com"
        self.siliconflow_base = "https://api.siliconflow.cn/v1"
        self.qwen_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.chatgpt_base = "https://api.openai.com/v1"
        self.volc_base = "https://ark.cn-beijing.volces.com/api/v3"  # Volcano Engine API address

        # Volcano Engine model ID mapping
        self.volc_model_mapping = {
            "deepseek-r1": "ep-",
            "deepseek-v3": "ep-",
            "deepseek-r1-distill-qwen-7b": "ep-",
            "deepseek-r1-distill-qwen-32b": "ep-"
        }

        # Silicon Flow model ID mapping
        self.siliconflow_model_mapping = {
            "Meta-Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "Meta-Llama-3.3-70B-Instruct": "meta-llama/Llama-3.3-70B-Instruct",
            "deepseek-r1-distill-qwen-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "deepseek-r1-distill-qwen-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "deepseek-r1-distill-qwen-14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "deepseek-r1-distill-llama-70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            "deepseek-r1-distill-llama-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "deepseek-v3": "deepseek-ai/DeepSeek-V3",
            "deepseek-r1": "deepseek-ai/DeepSeek-R1"
        }

        # Add model grouping information
        self.model_groups = {
            'bailian': {
                'deepseek': [
                    'deepseek-r1-distill-qwen-1.5b',
                    'deepseek-r1-distill-qwen-7b',
                    'deepseek-r1-distill-qwen-14b',
                    'deepseek-r1-distill-qwen-32b',
                    'deepseek-r1-distill-llama-70b',
                    'deepseek-r1-distill-llama-8b',
                    'deepseek-v3',
                    'deepseek-r1'
                ]
            },
            'siliconflow': {
                'deepseek': [
                    'deepseek-r1-distill-qwen-1.5b',
                    'deepseek-r1-distill-qwen-7b',
                    'deepseek-r1-distill-qwen-14b',
                    'deepseek-r1-distill-llama-70b',
                    'deepseek-r1-distill-llama-8b',
                    'deepseek-v3',
                    'deepseek-r1'
                ],
                'llama': ['llama-3.1-8b', 'llama-3.3-70b']
            },
            'volc': {
                'deepseek': ['deepseek-r1-distill-qwen-7b', 'deepseek-r1-distill-qwen-32b', 'deepseek-v3', 'deepseek-r1']
            }
        }

        # Model maximum tokens configuration
        self.model_max_tokens = {
            "qwen2.5-14b-instruct-1m": 8192,
            "qwen2.5-7b-instruct-1m": 8192,
            "qwen2.5-32b-instruct": 8192,
            "qwen2.5-14b-instruct": 8192,
            "qwen2.5-7b-instruct": 8192,
            "qwq-32b-preview": 16384,
            "gpt-4o": 16384,
            "gpt-4o-mini": 16384,
            "o1-mini": 65536,
            "deepseek-chat": 8192,
            "deepseek-reasoner": 8192,
            "deepseek-v3": 8192,
            "deepseek-r1": 32768,
            "deepseek-r1-distill-qwen-1.5b": 16384,
            "deepseek-r1-distill-qwen-7b": 16384,
            "deepseek-r1-distill-qwen-14b": 16384,
            "deepseek-r1-distill-llama-70b": 16384,
            "deepseek-r1-distill-llama-8b": 16384,
            "Meta-Llama-3.1-8B-Instruct": 4096,
            "Meta-Llama-3.3-70B-Instruct": 4096,
        }

        # Build SSL context
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE

        # HTTP client
        self.http_client = httpx.Client(verify=False)
        # Silicon Flow dedicated client - enable SSL verification
        self.siliconflow_client = requests.Session()
        self.siliconflow_client.verify = True  # Enable SSL verification

        # Set log level
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("urllib3").setLevel(logging.ERROR)
        logging.getLogger("openai").setLevel(logging.ERROR)

        # Initialize API call counters
        self.api_call_counts = {}
        self.last_call_time = {}

        # Set base path
        self.base_path = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))

        # Initialize clients
        self.setup_clients()

    def setup_clients(self):
        """Set up API clients for different providers"""
        try:
            from openai import OpenAI

            # Configure shorter timeout
            timeout = httpx.Timeout(
                timeout=360.0,    # Total timeout
                connect=360.0,     # Connection timeout
                read=360.0,       # Read timeout
                write=360.0        # Write timeout
            )

            # Configure httpx client
            transport = httpx.HTTPTransport(
                retries=3,       # Increase retry count
                verify=False,    # Disable SSL verification
                trust_env=False
            )

            self.http_client = httpx.Client(
                verify=False,    # Disable SSL verification
                timeout=timeout,
                transport=transport,
                trust_env=False
            )

            # Configure clients for different models
            self.qwen_client = OpenAI(
                api_key=self.qwen_key,
                base_url=self.qwen_base,
                http_client=self.http_client
            )

            self.deepseek_client = OpenAI(
                api_key=self.deepseek_key,
                base_url=self.deepseek_base,
                http_client=self.http_client
            )

            self.chatgpt_client = OpenAI(
                api_key=self.chatgpt_key,
                base_url=self.chatgpt_base,
                http_client=self.http_client
            )

            # Configure Volcano Engine client
            self.volc_client = OpenAI(
                api_key=self.volc_key,
                base_url=self.volc_base,
                http_client=self.http_client
            )

        except Exception as e:
            print(f"\nAPI client initialization failed: {str(e)}")
            raise

    def check_rate_limit(self, model_name: str):
        """Check and control API call rate"""
        import time

        # Initialize counter
        if model_name not in self.api_call_counts:
            self.api_call_counts[model_name] = 0
            self.last_call_time[model_name] = datetime.now()

        self.api_call_counts[model_name] += 1

        # Apply rate limiting for GPT series and o-series models
        if any(model_name.startswith(prefix) for prefix in ['gpt', 'o1']):
            if self.api_call_counts[model_name] % 10 == 0:
                time.sleep(3)

    def extract_answer(self, response: str) -> int:
        """Extract answer from model response"""
        if not response:
            return None

        # Find all boxed{number} patterns
        matches = list(re.finditer(r'boxed{(\d+)}', response))
        if matches:
            # Only take the last match
            last_match = matches[-1]
            return int(last_match.group(1))

        return None

    def call_model(self, model_name: str, polynomial: str, temperature: float, prompt_template: str, try_channels: List[str] = None) -> tuple:
        """Call model API, supporting multiple channels"""
        start_time = datetime.now()
        last_error = None

        def is_valid_answer(answer):
            """Validate if answer is valid"""
            return answer is not None and answer in [0, 1]

        def try_api_call(channel: str) -> Optional[tuple]:
            nonlocal start_time, last_error

            try:
                # Extract base model name from composite model name
                base_model_name = model_name.split('_')[0]
                self.check_rate_limit(base_model_name)
                prompt = prompt_template.replace("<INPUT>", polynomial)
                max_tokens = self.model_max_tokens.get(base_model_name, 4096)

                # Bailian channel (using Alibaba Cloud API)
                if channel == 'bailian' and base_model_name in self.model_groups['bailian']['deepseek']:
                    print(f"\nTrying via Bailian API - Model: {base_model_name}")
                    params = {
                        "model": base_model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "max_tokens": max_tokens
                    }
                    if temperature is not None:
                        params["temperature"] = temperature

                    time.sleep(3)  # Base waiting time

                    try:
                        response = self.qwen_client.chat.completions.create(
                            **params)
                        if response and hasattr(response, 'choices') and response.choices:
                            result = response.choices[0].message.content.strip(
                            )
                            answer = self.extract_answer(result)
                            if is_valid_answer(answer):
                                api_time = (datetime.now() -
                                            start_time).total_seconds()
                                return answer, result, api_time
                            else:
                                last_error = "Invalid answer format"
                                return None
                        else:
                            last_error = "API returned invalid response"
                            return None
                    except Exception as e:
                        last_error = str(e)
                        if "timeout" in str(e).lower() or "Model service timeout" in str(e):
                            time.sleep(10)  # Wait 10 seconds for timeout errors
                        return None

                # Silicon Flow channel
                elif channel == 'siliconflow' and base_model_name in self.siliconflow_model_mapping:
                    print(f"\nTrying via Silicon Flow API - Model: {base_model_name}")
                    params = {
                        "model": self.siliconflow_model_mapping[base_model_name],
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "max_tokens": max_tokens
                    }
                    if temperature is not None:
                        params["temperature"] = temperature

                    time.sleep(3)  # Increase base waiting time to 3 seconds

                    response = self.siliconflow_client.post(
                        f"{self.siliconflow_base}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.siliconflow_key}"},
                        json=params
                    )

                    if response.status_code == 200:
                        result = response.json()[
                            'choices'][0]['message']['content'].strip()
                        answer = self.extract_answer(result)
                        if is_valid_answer(answer):
                            api_time = (datetime.now() -
                                        start_time).total_seconds()
                            return answer, result, api_time
                    else:
                        last_error = f"API call failed: {response.text}"
                        if "timeout" in response.text.lower() or "Model service timeout" in response.text:
                            time.sleep(10)  # Wait 10 seconds for timeout errors
                        return None

                # Volcano Engine channel
                elif channel == 'volc' and base_model_name in self.volc_model_mapping:
                    print(f"\nTrying via Volcano Engine API - Model: {base_model_name}")
                    params = {
                        "model": self.volc_model_mapping[base_model_name],
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "max_tokens": max_tokens,
                        "temperature": temperature
                    }

                    time.sleep(3)  # Increase base waiting time to 3 seconds

                    response = self.volc_client.chat.completions.create(
                        **params)
                    if response and hasattr(response, 'choices') and response.choices:
                        result = response.choices[0].message.content.strip()
                        answer = self.extract_answer(result)
                        if is_valid_answer(answer):
                            api_time = (datetime.now() -
                                        start_time).total_seconds()
                            return answer, result, api_time
                        else:
                            last_error = "Invalid answer format"
                            return None
                    else:
                        last_error = "API returned invalid response"
                        return None

            except Exception as e:
                last_error = str(e)
                if "timeout" in str(e).lower() or "Model service timeout" in str(e):
                    time.sleep(10)  # Wait 10 seconds for timeout errors
                return None

            return None

        # Try all available channels
        channels = try_channels if try_channels else ['siliconflow', 'volc']
        max_retries = 3  # Maximum 3 retries per channel

        for channel in channels:
            for retry in range(max_retries):
                result = try_api_call(channel)
                if result is not None:
                    return result
                print(
                    f"\n{channel} channel call failed (retry {retry + 1}/{max_retries}): {last_error}")
                    time.sleep(3)  # Wait 3 seconds before each retry

        # Return error after all channels fail
        return None, f"All API channels failed, last error: {last_error}", None

    def process_file(self, input_data: Union[str, pd.DataFrame], model_configs: List[Dict], prompt_templates: Union[str, List[str]], rounds: int) -> pd.DataFrame:
        """Process input data"""
        try:
            start_time = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_dir = os.path.join(
                self.base_path, 'Analysis', f'Analysis_{timestamp}')
            os.makedirs(analysis_dir, exist_ok=True)

            # Prepare data
            if isinstance(input_data, str):
                df = pd.read_json(input_data, lines=True)
            else:
                df = input_data.copy()

            if df is None or len(df) == 0:
                print("[ERROR] Data is empty")
                return None

            # Ensure prompt_templates is a list
            if isinstance(prompt_templates, str):
                prompt_templates = [prompt_templates]

            # Get prompt_keys
            prompt_keys = self.experiment_config.get(
                'prompt_keys') if self.experiment_config else None
            if not prompt_keys:
                prompt_keys = [
                    f'prompt{i+1}' for i in range(len(prompt_templates))]

            # Create manager and queues
            manager = Manager()
            results_queue = manager.Queue(maxsize=5000)
            progress_queue = manager.Queue(maxsize=1000)

            # Create all task combinations
            tasks = []
            for row_idx, row in df.iterrows():
                for config in model_configs:
                    for prompt_template, prompt_key in zip(prompt_templates, prompt_keys):
                        tasks.append({
                            'row_idx': row_idx,
                            'row': row,
                            'model_config': config,
                            'prompt_template': prompt_template,
                            'prompt_key': prompt_key,
                            'try_channels': config.get('try_channels', ['volc'])
                        })

            # Group tasks by API provider
            grouped_tasks = {}
            for task in tasks:
                model_name = task['model_config']['name']
                for channel in task['try_channels']:
                    group_key = f"{channel}_{model_name}"
                    if group_key not in grouped_tasks:
                        grouped_tasks[group_key] = []
                    grouped_tasks[group_key].append(task)

            # Use thread pool to handle different groups of tasks
            max_workers = min(8, len(grouped_tasks))  # Maximum 8 concurrent groups
            results = []

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for group_name, group_tasks in grouped_tasks.items():
                    future = executor.submit(
                        self._process_task_group, group_tasks, results_queue, progress_queue)
                    futures.append((future, group_name))

                # Handle completed task groups
                for future, group_name in futures:
                    try:
                        future.result()
                    except Exception as e:
                        print(f"\nError processing task group {group_name}: {str(e)}")

                    # Process results queue
                    while True:
                        try:
                            result = results_queue.get_nowait()
                            results.append(result)

                            # Save current group results
                            self._save_group_results(
                                df, [result], analysis_dir, group_name)
                        except Empty:
                            break

                    # Process progress queue
                    while True:
                        try:
                            progress = progress_queue.get_nowait()
                            if 'error' not in progress:
                                print(f"\nTask progress update - {group_name}")
                                print(f"Model: {progress['model_prompt_key']}")
                                if 'stats' in progress:
                                    stats = progress['stats']
                                    if stats.get('result') is not None:
                                        print(f"Result: {stats['result']}")
                                    if stats.get('api_time'):
                                        print(
                                            f"Response time: {stats['api_time']:.1f}s")
                        except Empty:
                            break

            # Update final DataFrame
            for result in results:
                model_prompt_key = result['model_prompt_key']
                idx = result['idx']
                data = result['result']

                df.at[idx, f'{model_prompt_key}_output_1'] = data['result']
                df.at[idx,
                      f'{model_prompt_key}_raw_response_1'] = data['raw_response']
                df.at[idx, f'{model_prompt_key}_time_1'] = data['api_time']

            # Save final results
            predictions_file = os.path.join(analysis_dir, 'predictions.jsonl')
            df.to_json(predictions_file, orient='records', lines=True)
            df.to_csv(os.path.join(analysis_dir,
                      'predictions.csv'), index=False)

            return df

        except Exception as e:
            print(f"Error processing data: {str(e)}")
            raise

    def _process_task_group(self, tasks: List[dict], results_queue: Queue, progress_queue: Queue):
        """Process task group"""
        total_tasks = len(tasks)
        completed_tasks = 0
        group_name = f"{tasks[0]['try_channels'][0]}_{tasks[0]['model_config']['name']}"

        # Create progress bar
        bar_length = 50
        print(f"\nProcessing task group: {group_name}")
        print("-" * 50)

        def update_progress():
            """Update progress bar display"""
            percentage = float(completed_tasks) / float(total_tasks)
            filled_length = int(bar_length * percentage)
            bar = '=' * filled_length + '-' * (bar_length - filled_length)
            sys.stdout.write('\r')  # Move to line start
            sys.stdout.write(
                f"Progress [{bar}] {completed_tasks}/{total_tasks} ({percentage:.1%})")
            sys.stdout.flush()

        # Initial progress bar display
        update_progress()

        for task in tasks:
            try:
                result, raw_response, api_time = self.call_model(
                    task['model_config']['name'],
                    task['row']['polynomial'],
                    task['model_config']['temperature'],
                    task['prompt_template'],
                    task['try_channels']
                )

                # Process result
                prediction_result = {
                    'idx': task['row_idx'],
                    'result': result,
                    'raw_response': raw_response,
                    'api_time': api_time,
                    'input': task['row']['polynomial'],
                    'actual': task['row']['ans'],
                    'response': raw_response,
                    'is_correct': result == task['row']['ans'] if result is not None else None
                }

                # Send result and progress update
                model_prompt_key = f"{task['model_config']['name']}_{task['prompt_key']}"
                results_queue.put({
                    'model_prompt_key': model_prompt_key,
                    'idx': task['row_idx'],
                    'result': prediction_result
                })
                progress_queue.put({
                    'model_prompt_key': model_prompt_key,
                    'stats': prediction_result
                })

                # Display current task result
                print()  # New line, leave space for result
                if result is not None:
                    result_mark = '\033[92m✓\033[0m' if prediction_result['is_correct'] else '\033[91m✗\033[0m'
                    print(
                        f"Result: {result} | Actual: {task['row']['ans']} [{result_mark}] | Time: {api_time:.1f}s")
                else:
                    print("\033[91m[Call failed]\033[0m")

                # Update progress
                completed_tasks += 1
                update_progress()  # Update progress bar

                # API call interval
                if 'gpt' in task['model_config']['name'].lower() or 'o1-mini' in task['model_config']['name'].lower():
                    time.sleep(3)  # GPT series models need longer interval
                else:
                    time.sleep(0.5)  # Other models use shorter interval

            except Exception as e:
                print(f"\nError: {str(e)}")
                progress_queue.put({'error': str(e)})
                completed_tasks += 1
                update_progress()  # Update progress bar

        # Display 100% progress after completion
        print(f"\nTask group {group_name} completed\n")

    def _save_group_results(self, df: pd.DataFrame, results: List[dict], analysis_dir: str, group_name: str):
        """Save single task group results"""
        try:
            temp_df = df.copy()
            for result in results:
                model_prompt_key = result['model_prompt_key']
                idx = result['idx']
                data = result['result']

                temp_df.at[idx,
                           f'{model_prompt_key}_output_1'] = data['result']
                temp_df.at[idx,
                           f'{model_prompt_key}_raw_response_1'] = data['raw_response']
                temp_df.at[idx,
                           f'{model_prompt_key}_time_1'] = data['api_time']

            group_file = os.path.join(
                analysis_dir, f'predictions_{group_name}.jsonl')
            temp_df.to_json(group_file, orient='records', lines=True)

        except Exception as e:
            print(f"Error saving {group_name} group results: {str(e)}")


def process_single_task(task: dict, results_queue: Queue, progress_queue: Queue):
    """Process single task"""
    try:
        api_caller = APIConfig()
        api_caller.setup_clients()

        row_idx = task['row_idx']
        row = task['row']
        config = task['model_config']
        prompt_template = task['prompt_template']
        prompt_key = task['prompt_key']

        model_prompt_key = f"{config['name']}_{prompt_key}"

        max_retries = 3
        retry_count = 0
        success = False

        while retry_count < max_retries and not success:
            try:
                result, raw_response, api_time = api_caller.call_model(
                    config['name'],
                    row['polynomial'],
                    config['temperature'],
                    prompt_template
                )

                # Process result
                prediction_result = {
                    'idx': row_idx,
                    'result': result,
                    'raw_response': raw_response,
                    'api_time': api_time,
                    'input': row['polynomial'],
                    'actual': row['ans'],
                    'response': raw_response,
                    'is_correct': result == row['ans'] if result is not None else None
                }

                # Send result
                results_queue.put({
                    'model_prompt_key': model_prompt_key,
                    'idx': row_idx,
                    'result': prediction_result
                })

                # Send progress update
                progress_queue.put({
                    'model_prompt_key': model_prompt_key,
                    'stats': prediction_result
                })

                success = True

            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"\n[Retry {retry_count}/{max_retries}] {str(e)}")
                    time.sleep(2 ** retry_count)  # Exponential backoff
                    continue

                # Send error result
                error_result = {
                    'idx': row_idx,
                    'result': None,
                    'raw_response': f"Error: {str(e)}",
                    'api_time': None,
                    'input': row['polynomial'],
                    'actual': row['ans'],
                    'response': f"Error: {str(e)}",
                    'is_correct': None
                }

                results_queue.put({
                    'model_prompt_key': model_prompt_key,
                    'idx': row_idx,
                    'result': error_result
                })

                progress_queue.put({
                    'model_prompt_key': model_prompt_key,
                    'stats': error_result
                })

        # API call interval
        if 'gpt' in config['name'].lower() or 'o1-mini' in config['name'].lower():
            time.sleep(3)  # GPT series models need longer interval
        else:
            time.sleep(0.5)  # Other models use shorter interval

    except Exception as e:
        print(f"\nError processing task: {str(e)}")
        progress_queue.put({
            'error': f"Task processing error: {str(e)}"
        })


class SOSPredictor:
    """SOS Predictor Class"""

    def __init__(self):
        self.api_config = APIConfig()
        self.experiment_config = None

    def process_file(self, input_data: Union[str, pd.DataFrame], model_configs: List[Dict], prompt_templates: Union[str, List[str]], rounds: int) -> pd.DataFrame:
        """Process input data"""
        # Ensure experiment_config is passed to APIConfig
        if hasattr(self, 'experiment_config'):
            self.api_config.experiment_config = self.experiment_config
        return self.api_config.process_file(input_data, model_configs, prompt_templates, rounds)


if __name__ == "__main__":
    predictor = SOSPredictor()

    # Configure parameters
    input_file = 'evaluation.jsonl'
    model_configs = [
        {'name': 'o1-mini', 'temperature': 0.1}
    ]
    prompt_templates = "Is the following expression a Sum of Squares (SOS) polynomial? {polynomial}. Return 1 if it is, otherwise return 0."
    rounds = 1

    # Process file
    predictor.process_file(input_file, model_configs, prompt_templates, rounds)
