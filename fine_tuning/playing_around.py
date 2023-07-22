from typing import Final

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Conversation,
    PreTrainedModel,
    PreTrainedTokenizer,
)


MODEL_NAME: Final[str] = "facebook/blenderbot_small-90M"


def _load_model(model_name: str = MODEL_NAME) -> PreTrainedModel:
    return AutoModelForSeq2SeqLM.from_pretrained(model_name)


def _load_tokenizer(model_name: str = MODEL_NAME) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(model_name)


def _prepare_conversation_history(conversation: Conversation) -> str:
    prepped_history = [
        f"{'<s>' if idx > 0 else ''} {text}" if is_user else f"<s>{text}"
        for idx, (is_user, text) in enumerate(conversation.iter_texts())
    ]
    return "</s> ".join(prepped_history)


def _display_conversation(conversation: Conversation) -> None:
    print(f"{'~' * 29}CONVERSATION:{'~' * 29}\n")
    for is_user, text in conversation.iter_texts():
        print(f"{'User: ' if is_user else 'Bot: '}{text}")
    print(f"{'~' * 71}\n")


def learning_how_to_interact_with_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
) -> None:
    conversation = Conversation()

    # We're just going in with our questions regardless of responses, to hell with the AI's responses
    user_inputs = [
        "Hello, how are you today?",
        "Ah yeah that is nice. What are your hobbies?",
    ]
    for user_input in user_inputs:
        print(f"\n>Generating response for '{user_input}'...\n")
        conversation.add_user_input(user_input)
        conversation_history = _prepare_conversation_history(conversation)
        prepared_model_input = tokenizer([conversation_history], return_tensors="pt")
        generated_response = model.generate(**prepared_model_input, max_new_tokens=128)[0]
        conversation.mark_processed()
        conversation.append_response(
            tokenizer.decode(generated_response, skip_special_tokens=True)
        )
        _display_conversation(conversation)


if __name__ == "__main__":
    model = _load_model()
    tokenizer = _load_tokenizer()
    learning_how_to_interact_with_model(model, tokenizer)

