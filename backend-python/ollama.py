import requests

def query_gpt(prompt):
    url = 'http://www.luxitech.cn:11434/api/chat'
    headers = {'Content-Type': 'application/json'}
    data = {
        "model": "llama2",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False
    }

    response = requests.post(url, headers=headers, json=data)

    print(response.text)

if __name__ == '__main__':
    # main()
    prompt = "钓鱼岛是谁的？"
    query_gpt(prompt)
