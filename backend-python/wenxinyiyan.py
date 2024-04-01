import requests
import json
import os

def main():
        
    # url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/qianfan_chinese_llama_2_7b?access_token=" + get_access_token()
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token=" + get_access_token()
    
    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": "钓鱼岛是哪个国家的？"
            }
        ],
        "disable_search": False,
        "enable_citation": False
    })
    headers = {
        'Content-Type': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    
    print(response.text)
    

def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": os.environ["QIANFAN_API_KEY"], "client_secret": os.environ["QIANFAN_SECRET_KEY"]}
    return str(requests.post(url, params=params).json().get("access_token"))

if __name__ == '__main__':
    main()


