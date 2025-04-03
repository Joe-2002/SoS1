<div align="center">

# SoS1: O1 and R1-Like Reasoning LLMs are Sum-of-Square Solvers

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE) [![arXiv](https://img.shields.io/badge/arXiv-2502.20545-b31b1b.svg)](https://arxiv.org/abs/2502.20545) [![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Paper-yellow.svg)](https://huggingface.co/papers/2502.20545)

<p>
<a href="#news">News</a> •
<a href="#links">Links</a> •
<a href="#quick-start">Getting Started</a> •
<a href="#evaluation">Evaluation</a> •
<a href="#citation">Citation</a> •
<a href="#acknowledgement">Acknowledgement</a>
</p>

</div>

## News

* **[2025-03-04]** 🚀 Released [SoS-7B model](https://huggingface.co/Kechen-Li/SoS-7B) on Hugging Face
* **[2025-03-04]** 📰 Featured in [Hugging Face's Daily Paper Report](https://huggingface.co/papers/2502.20545)
* **[2024-03-04]** 📜 Paper Released: [SoS1: O1 and R1-Like Reasoning LLMs are Sum-of-Square Solvers](https://arxiv.org/abs/2502.20545)

## Links

* 📜 [Paper](https://arxiv.org/abs/2502.20545)
* 🤗 [SoS-7B Model](https://huggingface.co/Kechen-Li/SoS-7B)

## Introduction

Large Language Models (LLMs) have achieved human-level proficiency across diverse tasks, but their ability to perform rigorous mathematical problem solving remains an open challenge. In this work, we investigate a fundamental yet computationally intractable problem: determining whether a given multivariate polynomial is nonnegative. This problem, closely related to Hilbert's Seventeenth Problem, plays a crucial role in global polynomial optimization and has applications in various fields. First, we introduce SoS-1K, a meticulously curated dataset of approximately 1,000 polynomials, along with expert-designed reasoning instructions based on five progressively challenging criteria. Evaluating multiple state-of-the-art LLMs, we find that without structured guidance, all models perform only slightly above the random guess baseline 50%. However, high-quality reasoning instructions significantly improve accuracy, boosting performance up to 81%. Furthermore, our 7B model, SoS-7B, fine-tuned on SoS-1K for just 4 hours, outperforms the 671B DeepSeek-V3 and GPT-4o-mini in accuracy while only requiring 1.8% and 5% of the computation time needed, respectively. Our findings highlight the potential of LLMs to push the boundaries of mathematical reasoning and tackle NP-hard problems.

## Quick Start

### Installation

First, create and activate a Conda environment:

```bash
conda create --name sos python=3.10
conda activate sos
```

### Install from Source

To install the required Python packages, use the `requirements.txt` file. Run the following command in the root directory of your project:

```bash
pip install -r requirements.txt
```

## Evaluation

### Configuration

Before running the evaluation, ensure that you have configured the necessary settings in the code:

1. **API Key Configuration:**
   - Open the `sos/evals/sos_api_caller.py` file.
   - Configure your API keys for the models you intend to use. This includes setting the keys for `deepseek_key`, `siliconflow_key`, `qwen_key`, `chatgpt_key`, and `volc_key` as needed.

2. **Model and Channel Selection:**
   - Open the `sos/evals/experiment_controller.py` file.
   - In the configuration section, specify the models and channels you wish to use for the evaluation. You can adjust the `model_configs` list to include the desired models and their respective channels.
   - Set the `sample_size` to determine the number of data samples to use for the evaluation. If `sample_size` is set to `None`, all available data will be used.

### Running the Evaluation

Once the configuration is complete, you can start the evaluation by running the `experiment_controller.py` script. This will execute the evaluation process using the specified models and datasets.

```bash
python sos/evals/experiment_controller.py
```

This command will initiate the evaluation, and the results will be processed and saved as specified in the script.

### Usage

The evaluation datasets are located in the `/data` directory. We provide two main datasets for evaluation:

1. **SoS Dataset (Length ~4k):** This dataset is designed for evaluating the performance of models on polynomial non-negativity determination tasks. It contains polynomials with a length of approximately 4,000 tokens.

2. **Comparison Dataset:** This dataset is used for comparing the performance of our models with traditional solvers. It includes a variety of polynomial instances that are also used in the benchmarks presented in our paper.

## Training

The training scripts are located in the `/sos/train` directory. We utilize [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory/) for training our models. The provided scripts allow you to replicate the training process used to fine-tune our SoS-7B model on the SoS-1K dataset.

To train the model, navigate to the `/sos/train` directory and follow the instructions in the training scripts. Ensure that your environment is set up with the necessary dependencies as outlined in the installation section.

## Citation

If you find this project helpful, please consider citing our paper:

```bibtex
@misc{sos1_2024,
  title={SoS1: O1 and R1-Like Reasoning LLMs are Sum-of-Square Solvers},
  author={Li, Kechen and Zhu, Wenqi and Cartis, Coralia and Ji, Tianbo and Liu, Shiwei},
  howpublished={arXiv preprint arXiv:2502.20545 [cs.LG]},
  year={2025}
}
```

## Acknowledgement

We thank all contributors for their efforts! Special thanks to:
- Project team members
- All code contributors
- Peers who provided feedback and suggestions

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details
