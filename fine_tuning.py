import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

class ResumeAnalyzer:
    """
    A class to encapsulate resume analysis using the Unsloth library and LLMs.
    """

    def __init__(self, model_name, max_seq_length=2048):
        """
        Initialize the ResumeAnalyzer with a pre-trained model and tokenizer.

        Args:
            model_name (str): Name of the pre-trained model to load.
            max_seq_length (int): Maximum sequence length for the tokenizer.
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """
        Load the model and tokenizer using the Unsloth library.
        """
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )
            print("Model and tokenizer loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")

    def optimize_model(self, r=16, target_modules=None, lora_alpha=16, lora_dropout=0.05, bias="none", use_gradient_checkpointing=True, use_rslora=True):
        """
        Apply PEFT (Parameter-Efficient Fine-Tuning) optimizations to the model.

        Args:
            r (int): Rank value for low-rank adaptation.
            target_modules (list): List of target modules to optimize.
            lora_alpha (int): Scaling factor for LoRA.
            lora_dropout (float): Dropout probability for LoRA layers.
            bias (str): Bias configuration for LoRA.
            use_gradient_checkpointing (bool): Enable gradient checkpointing.
            use_rslora (bool): Enable Randomized SVD LoRA.
        """
        if not target_modules:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        try:
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=r,
                target_modules=target_modules,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=bias,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_rslora=use_rslora
            )
            print("Model optimization completed.")
        except Exception as e:
            print(f"Error optimizing model: {e}")

    @staticmethod
    def load_dataset(dataset_name="cnamuangtoun/resume-job-description-fit", label_map=None):
        """
        Load and preprocess the dataset for training and evaluation.

        Args:
            dataset_name (str): Name of the dataset to load.
            label_map (dict): Mapping of textual labels to numerical labels.

        Returns:
            DatasetDict: Preprocessed dataset.
        """
        if label_map is None:
            label_map = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}

        def preprocess_function(examples):
            inputs = [f"Resume: {resume}\nJob Description: {job}\nFit:" for resume, job in zip(examples["resume_text"], examples["job_description_text"])]
            labels = [label_map[label] for label in examples["label"]]
            return {"text": inputs, "label": labels}

        try:
            dataset = load_dataset(dataset_name)
            tokenized_train = dataset["train"].map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
            tokenized_test = dataset["test"].map(preprocess_function, batched=True, remove_columns=dataset["test"].column_names)
            print("Dataset loaded and preprocessed successfully.")
            return {"train": tokenized_train, "test": tokenized_test}
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    @staticmethod
    def evaluate(predictions, labels):
        """
        Evaluate the model using precision, recall, F1-score, and accuracy.

        Args:
            predictions (list): List of predicted labels.
            labels (list): List of true labels.

        Returns:
            dict: Evaluation metrics.
        """
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
        accuracy = accuracy_score(labels, predictions)

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
        }

    def fine_tune(self, dataset, output_dir="./fine_tuned_model", epochs=3, learning_rate=5e-5, warmup_steps=500):
        """
        Fine-tune the model on the provided dataset.

        Args:
            dataset (dict): Preprocessed dataset with "train" and "test" splits.
            output_dir (str): Directory to save the fine-tuned model.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for the optimizer.
            warmup_steps (int): Number of warmup steps for the scheduler.
        """
        if self.model is None or self.tokenizer is None:
            print("Model and tokenizer must be loaded before fine-tuning.")
            return

        try:
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=4,
                save_steps=10_000,
                save_total_limit=2,
                logging_dir=f"{output_dir}/logs",
                learning_rate=learning_rate,
                warmup_steps=warmup_steps
            )

            trainer = SFTTrainer(
                model=self.model,
                train_dataset=dataset["train"],
                eval_dataset=dataset["test"],
                tokenizer=self.tokenizer,
                args=training_args,
            )

            trainer.train()
            print("Fine-tuning completed.")

            # Evaluate model
            predictions = trainer.predict(dataset["test"]).predictions
            predicted_labels = predictions.argmax(axis=-1)
            metrics = self.evaluate(predicted_labels, dataset["test"]["label"])
            print(f"Evaluation Metrics: {metrics}")
        except Exception as e:
            print(f"Error during fine-tuning: {e}")

# Example Usage
if __name__ == "__main__":
    # LLaMA 3.1
    llama_3_1_analyzer = ResumeAnalyzer("unsloth/Meta-Llama-3.1-8B-bnb-4bit")
    llama_3_1_analyzer.load_model()
    llama_3_1_analyzer.optimize_model()

    data = llama_3_1_analyzer.load_dataset()
    if data:
        llama_3_1_analyzer.fine_tune(data)

    # LLaMA 2
    llama_2_analyzer = ResumeAnalyzer("unsloth/LLaMA-2-7B-bnb-4bit")
    llama_2_analyzer.load_model()
    llama_2_analyzer.optimize_model()

    data = llama_2_analyzer.load_dataset()
    if data:
        llama_2_analyzer.fine_tune(data)

    # Mistral
    mistral_analyzer = ResumeAnalyzer("unsloth/Mistral-7B-bnb-4bit")
    mistral_analyzer.load_model()
    mistral_analyzer.optimize_model()

    data = mistral_analyzer.load_dataset()
    if data:
        mistral_analyzer.fine_tune(data)
