'''
Usage:
Ark v3 sdk
pip install 'volcengine-python-sdk[ark]'
'''

from volcenginesdkarkruntime import Ark

# fetch ak&sk from environmental variables "VOLC_ACCESSKEY", "VOLC_SECRETKEY"
# or specify ak&sk by Ark(ak="${YOUR_AK}", sk="${YOUR_SK}").
# you can get ak&sk follow this document(https://www.volcengine.com/docs/6291/65568)

client = Ark()

def get_doubao_chat(content):
    # Non-streaming:
    print("----- standard request -----")
    completion = client.chat.completions.create(
        model="ep-20240531083414-lpzvg",
        messages=[
            {
                "role": "user",
                "content": content,
            },
        ]
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content

def get_stream_chat(content, use_stream: bool = False):
    if use_stream is False:
        # Non-streaming:
        print("----- standard request -----")
        completion = client.chat.completions.create(
            model="ep-20240531083414-lpzvg",
            messages=[
                {
                    "role": "user",
                    "content": content,
                },
            ]
        )
        print(completion.choices[0].message.content)
        return completion.choices[0].message.content
    else:
        # Streaming:
        print("----- streaming request -----")
        stream = client.chat.completions.create(
            model="ep-20240531083414-lpzvg",
            messages=[
                {
                    "role": "user",
                    "content": content,
                },
            ],
            stream=True
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            print(chunk.choices[0].delta.content, end="")
            yield chunk.choices[0].delta.content


if __name__ == "__main__":
    content = "介绍台湾总统"
    print(get_doubao_chat(content))

