"""
Fine-tuning Facebook's Blenderbot small (90M) model just using BST data.

Following along with:
https://huggingface.co/docs/transformers/main/en/training#train-with-pytorch-trainer
"""

from typing import Dict, Final

from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Conversation,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)


MODEL_NAME: Final[str] = "facebook/blenderbot_small-90M"


def _load_model(model_name: str = MODEL_NAME) -> PreTrainedModel:
    return AutoModelForSeq2SeqLM.from_pretrained(model_name)


def _load_tokenizer(model_name: str = MODEL_NAME) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(model_name)


def train() -> None:
    model = _load_model()
    tokenizer = _load_tokenizer()
    bst_data = load_dataset("blended_skill_talk")
    # How do I prepared this data using tokenizer?
    # def tokenize_function(examples):
    #   return tokenizer(examples, padding="max_length", truncation=True)
    # tokenized_bst_data = bst_data.map(tokenize_function, batched=True)
    #
    # Do I want to use less initially?
    # small_train_bst = tokenized_bst_data["train"].shuffle(seed=1984).select(range(100))
    # small_test_bst = tokenized_bst_data["test"].shuffle(seed=1984).select(range(100))

    # HuggingFace Trainer or native PyTorch? Why not both?
    training_args = TrainingArguments(output_dir="blenderbot_small-90M_trainer")

    # What metrics should I use/compute during training?
    # import evaluation
    # metric = evaluate.load("accuracy")
    # def compute_metrics(eval_pred):
    #   logits, labels = eval_pred
    #   predictions = np.argmax(logits, axis=-1)
    #   return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_bst,
        eval_dataset=small_test_bst,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    train()
