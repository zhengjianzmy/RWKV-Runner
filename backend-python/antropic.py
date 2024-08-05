import anthropic
import os


api_key=os.environ.get("ANTHROPIC_API_KEY")

print(api_key)


client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    # api_key="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",

    # api_key="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",
    api_key="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",
    # api_key = api_key,
)



message = client.messages.create(
    model="claude-3-5-sonnet-20240620",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Hello, Claude"
        }
    ],
    # model="claude-3-opus-20240229"
)
print(message.content)
