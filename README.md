This repository provides a modular framework for fine-tuning and evaluating state-of-the-art language models (e.g., LLaMA, Mistral) for resume and job description matching using the Unsloth library.

---

## Features
- **Model Support**: Fine-tune pre-trained models like LLaMA-3.1, LLaMA-2, and Mistral using parameter-efficient techniques.
- **Optimization**: Employ LoRA-based optimizations for memory-efficient training.
- **Evaluation**: Measure performance using precision, recall, F1-score, and accuracy.
- **Dataset Handling**: Load and preprocess datasets for resume-job description fit analysis.

---
## Requirements:
- Python 3.8+
- GPU with CUDA support for model fine-tuning



## Installation

1. Clone the repository: git clone https://github.com/Ruochen1105/IDLS_Fall2024.git
2.  cd IDLS_Fall2024

## Running the Code
1.  Load, finetune, run and evaluate the model: python Fine\ Tuning.py. Customize model parameters: Modify parameters like epochs, learning_rate, and warmup_steps in the fine_tune method.
2.  The default dataset used is cnamuangtoun/resume-job-description-fit. You can replace this with another dataset by modifying the load_dataset method.

For prompt engineering and using fine-tuning with prompt engineering, follow the guidelines mentioned in the respective jupyter notebooks.

Dataset_Analysis.ipynb contains code to analyze and extract statistics from the dataset.

## Results and Analysis
#### Reproducibility
 - Use the same random seeds and hyperparameters to reproduce results.
#### Metrics
Evaluate model performance with:
1. Precision
2. Recall
3. F1-score
4. Accuracy
