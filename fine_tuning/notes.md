# Notes

- The tokenizer/embedding has a vocabulary size of 54,944 and it uses Byte-Pair Encoding
    - Look more into "Byte-Pair encoding"
- The tokenizer warns the user when the maximum sequence length of the encoded text is greater than the 512 expected by the model's input.
    - I wonder what the best solution to this would be if we wanted to utilize this in a chatbot while maintaining some amount of conversational history.
Just use a sliding window to select the most recent?
- To interact with the model,
first tokenize the conversation (and history) while ensuring the returned object is a tensor,
then generate a response from the model (setting maximum new tokens to 128 for a default),
and finally decode the response (provided in a nested structure) while skipping the special tokens provided by the model to show the start and end of messages.
- Ran into a serious issue with the 90M model,
once I started including conversational history the model response just echoed back part of the history.
I ensured it was not implementation related by using the ConversationalPipeline from transformers.
Additionally,
I loaded the 400M-distill model and it worked as expected.
