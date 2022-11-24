import os
import openai
openai.api_key = "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"


def uploadClassificationDocument():
    fileName = "sentiment_training.jsonl"
    response = openai.File.create(file=open(fileName), purpose="classifications")
    return response

uploadResponse = uploadClassificationDocument()
fileName = uploadResponse.id
print(fileName)
response = openai.Classification.create(
    file=fileName,
    query="should buy bitcoin",
    search_model="ada", 
    model="curie", 
    max_examples=3
)
print(response)
# response = openai.Completion.create(
# engine="davinci",
# prompt="Blog topics dealing with daily life living on Mars\n\n1.",
# temperature=0.3,
# max_tokens=64,
# top_p=1,
# frequency_penalty=0.5,
# presence_penalty=0)
# print(response)