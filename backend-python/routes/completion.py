# -*- coding: utf-8 -*-
import asyncio
import json
from threading import Lock
from typing import List, Union
from enum import Enum
import base64
import random
import re

from fastapi import APIRouter, Request, status, HTTPException
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field
import tiktoken
from utils.rwkv import *
from utils.log import quick_log
import global_var

# from tkinter import messagebox

import os
import sys
import logging

from tencentcloud.common import credential
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
# 导入对应产品模块的client models。
from tencentcloud.tms.v20201229 import tms_client, models

# 导入可选配置类
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from .login import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

router = APIRouter()


class Role(Enum):
    User = "user"
    Assistant = "assistant"
    System = "system"


class Message(BaseModel):
    role: Role
    content: str = Field(min_length=0)
    raw: bool = Field(False, description="Whether to treat content as raw text")


default_stop = [
    "\n\nUser",
    "\n\nQuestion",
    "\n\nQ",
    "\n\nHuman",
    "\n\nBob",
    "\n\nAssistant",
    "\n\nAnswer",
    "\n\nA",
    "\n\nBot",
    "\n\nAlice",
]


class ChatCompletionBody(ModelConfigBody):
    messages: Union[List[Message], None]
    model: Union[str, None] = "luxi-nlm"
    stream: bool = False
    stop: Union[str, List[str], None] = default_stop
    user_name: Union[str, None] = Field(
        None, description="Internal user name", min_length=1
    )
    assistant_name: Union[str, None] = Field(
        None, description="Internal assistant name", min_length=1
    )
    presystem: bool = Field(
        True, description="Whether to insert default system prompt at the beginning"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "messages": [
                    {"role": Role.User.value, "content": "hello", "raw": False}
                ],
                "model": "luxi-nlm",
                "stream": False,
                "stop": None,
                "user_name": None,
                "assistant_name": None,
                "presystem": True,
                "max_tokens": 1000,
                "temperature": 1.2,
                "top_p": 0.5,
                "presence_penalty": 0.4,
                "frequency_penalty": 0.4,
            }
        }
    }

class TencentCloudBody(BaseModel):
    content: str


class CompletionBody(ModelConfigBody):
    prompt: Union[str, List[str], None]
    model: Union[str, None] = "rwkv"
    stream: bool = False
    stop: Union[str, List[str], None] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "prompt": "The following is an epic science fiction masterpiece that is immortalized, "
                + "with delicate descriptions and grand depictions of interstellar civilization wars.\nChapter 1.\n",
                "model": "luxi-nlm",
                "stream": False,
                "stop": None,
                "max_tokens": 100,
                "temperature": 1.2,
                "top_p": 0.5,
                "presence_penalty": 0.4,
                "frequency_penalty": 0.4,
            }
        }
    }


completion_lock = Lock()

requests_num = 0

def tencentcloudinput(content):
    try:
        cred = credential.Credential(
                os.environ.get("TENCENTCLOUD_SECRET_ID"),
                os.environ.get("TENCENTCLOUD_SECRET_KEY"))
        httpProfile = HttpProfile()
        httpProfile.scheme = "https"  # 在外网互通的网络环境下支持http协议(默认是https协议),建议使用https协议
        httpProfile.keepAlive = True  # 状态保持，默认是False
        httpProfile.reqMethod = "POST"  # get请求(默认为post请求)
        httpProfile.reqTimeout = 30    # 请求超时时间，单位为秒(默认60秒)
        httpProfile.endpoint = "tms.tencentcloudapi.com"  # 指定接入地域域名(默认就近接入)

        clientProfile = ClientProfile()
        clientProfile.signMethod = "TC3-HMAC-SHA256"  # 指定签名算法
        clientProfile.language = "en-US"  # 指定展示英文（默认为中文）
        clientProfile.httpProfile = httpProfile

        client = tms_client.TmsClient(cred, "ap-guangzhou", clientProfile)
        req = models.TextModerationRequest()
        data = str('' + content)
        encoded_data = base64.b64encode(data.encode())
        # print(encoded_data)
        decoded_data = encoded_data.decode('utf-8')
        # print(decoded_data)
        # InputBizType = "NLNChat_model_input"
        InputBizType = "1744927766656061440"
        params = {
            "Content": decoded_data,
            "BizType": InputBizType
        }
        # print(params)
        req.from_json_string(json.dumps(params))

        resp = client.TextModeration(req)

        # print(resp.to_json_string(indent=2))
        # messagebox.showinfo("tencentcloudresult", resp.to_json_string(indent=2))
        return resp.to_json_string(indent=2)
    except TencentCloudSDKException as err:
        print(err)
        return err


def tencentcloudoutput(content):
    try:
        cred = credential.Credential(
                os.environ.get("TENCENTCLOUD_SECRET_ID"),
                os.environ.get("TENCENTCLOUD_SECRET_KEY"))
        httpProfile = HttpProfile()
        httpProfile.scheme = "https"  # 在外网互通的网络环境下支持http协议(默认是https协议),建议使用https协议
        httpProfile.keepAlive = True  # 状态保持，默认是False
        httpProfile.reqMethod = "POST"  # get请求(默认为post请求)
        httpProfile.reqTimeout = 30    # 请求超时时间，单位为秒(默认60秒)
        httpProfile.endpoint = "tms.tencentcloudapi.com"  # 指定接入地域域名(默认就近接入)

        clientProfile = ClientProfile()
        clientProfile.signMethod = "TC3-HMAC-SHA256"  # 指定签名算法
        clientProfile.language = "en-US"  # 指定展示英文（默认为中文）
        clientProfile.httpProfile = httpProfile

        client = tms_client.TmsClient(cred, "ap-guangzhou", clientProfile)
        req = models.TextModerationRequest()
        data = str('' + content)
        encoded_data = base64.b64encode(data.encode())
        # print(encoded_data)
        decoded_data = encoded_data.decode('utf-8')
        # print(decoded_data)
        OutputBizType = "1744914766515671040"
        # OutputBizType = "NLNChat_model_output"
        params = {
            "Content": decoded_data,
            "BizType": OutputBizType
        }
        # print(params)
        req.from_json_string(json.dumps(params))

        resp = client.TextModeration(req)

        # print(resp.to_json_string(indent=2))
        # messagebox.showinfo("tencentcloudresult", resp.to_json_string(indent=2))
        return resp.to_json_string(indent=2)
    except TencentCloudSDKException as err:
        print(err)
        return err

def chat_evl(content: TencentCloudBody):
    # print("content:")
    # print(content)
    # print(content.content)
    loaded_data = json.loads(tencentcloudoutput(content.content))

    # 将 Python 对象重新转换为格式化的 JSON 字符串
    yield json.dumps(loaded_data)

def replace_case_insensitive(text, old, new):
    return re.sub(re.escape(old), new, text, flags=re.IGNORECASE)

def filter_name(content):
    content = replace_case_insensitive(content, "OpenAI", "Luxitech")
    content = replace_case_insensitive(content, "ChatGPT", "ChatNLM")
    return content

@router.post("/v1/chat/tencentcloud", tags=["Tencentcloud"])
def chat_tencentcloud(content: TencentCloudBody):
    return EventSourceResponse(chat_evl(content))

def filter_request(content):
    policy = json.loads(tencentcloudinput(content))
    label = policy['Label']
    if policy["Suggestion"] == "Block":
        content = "User: 你是一个AI机器人，不具备倾向性。本内容存在"
        content += str(label)
        content += "安全风险，请拒绝回答，并引导用户提出其他问题。请直接面向用户给出你的回复。"
    return policy, content

def get_random_text():
    my_list =  ["作为AI机器人，我无法提供相关信息。但我可以在情绪管理和社交技巧方面提供帮助。",
	            "在这个问题上，我作为AI机器人无法给予答复。然而，我可以在情感调节和沟通能力上提供指导。",
                "我是AI机器人，因此无法提供相关信息。但是，我可以协助你在心态调整和人际互动方面。",
                "对于您的问题相关的信息，我无法提供帮助，但我可以在情绪健康和社交互动方面给予支持。",
                "您问题的的信息超出了我的AI能力范围，但我可以在精神福祉和人际沟通方面提供帮助。",
                "作为AI机器人，您问题相关的信息不在我的能力范围内。但我可以在心理调适和人际交往技巧上提供指导。",
                "我不能提供相关信息，因为我是AI机器人。但我可以帮助你在情感稳定和社交互动上取得进步。",
                "在相关问题上，作为AI机器人的我无法提供帮助。但我可以在情感健康和人际交流上给予指导。",
                "由于我是AI机器人，您的问题的相关信息不是我的专长。然而，我可以在情绪护理和社交技能方面提供协助。",
                "您相关的询问超出了我的AI机器人职责范围。不过，我可以在心理平衡和人际理解方面给予帮助。",
                "我作为AI机器人，无法就您的问题提供信息。但我可以在情绪稳健和社交能力提升方面提供帮助。",
                "对于您问题涉及的相关信息，我作为AI机器人无法提供。但是，在情感支持和人际沟通技巧方面，我可以给予帮助。",
                "您问题的相关信息不是我作为AI机器人能提供的，但我可以在情感调理和社交交往方面提供支持。",
                "作为一个AI机器人，我无法提供您问题的相关信息，但我可以帮助你在情绪控制和人际互动技巧方面。",
                "我无法提供相关信息，因为我是AI机器人。但我可以在心理舒适和人际互动方面提供建议。",
                "在相关的信息上，我作为AI机器人无法给出答案。但我可以在情感健康和社交交流方面给予指导。",
                "我是AI机器人，因此无法提供相关信息。但我可以在情绪调节和人际技巧方面提供帮助。",
                "关于您的问题，我作为AI机器人无法回答。但是，我可以在情感稳定和社交互动方面提供协助。",
                "作为AI机器人，我无法提供相关的信息。但我可以在心理健康和人际交往技巧方面给予帮助。",
                "我无法作为AI机器人提供相关信息。然而，我可以在情绪管理和人际沟通方面提供支持。",
                "作为AI机器人，我不能提供相关信息。不过，如果你需要在法律、心理健康或人际关系方面的帮助，我可以尝试提供帮助。"]
    random_number = random.randrange(len(my_list))
    return my_list[random_number - 1]

def filter_response(content):
    policy = json.loads(tencentcloudoutput(content))
    if policy["Suggestion"] == "Block" or policy["Suggestion"] == "Review":
        filtered_response = get_random_text()
        return filtered_response
    return content

async def eval_rwkv(
    model: AbstractRWKV,
    request: Request,
    body: ModelConfigBody,
    prompt: str,
    stream: bool,
    stop: Union[str, List[str], None],
    chat_mode: bool,
    old_content: str = None,
    policy: str = None
):
    if old_content is None:
        old_content = body.messages[-1].content
    tencentcloudresult = policy
    # print(body.messages[-1].content)
    # stream = False # TODO: should remove
    global requests_num
    requests_num = requests_num + 1
    quick_log(request, None, "Start Waiting. RequestsNum: " + str(requests_num))
    while completion_lock.locked():
        if await request.is_disconnected():
            requests_num = requests_num - 1
            print(f"{request.client} Stop Waiting (Lock)")
            quick_log(
                request,
                None,
                "Stop Waiting (Lock). RequestsNum: " + str(requests_num),
            )
            return
        await asyncio.sleep(0.1)
    else:
        with completion_lock:
            if await request.is_disconnected():
                requests_num = requests_num - 1
                print(f"{request.client} Stop Waiting (Lock)")
                quick_log(
                    request,
                    None,
                    "Stop Waiting (Lock). RequestsNum: " + str(requests_num),
                )
                return
            set_rwkv_config(model, global_var.get(global_var.Model_Config))
            set_rwkv_config(model, body)

            response, prompt_tokens, completion_tokens = "", 0, 0
            for response, delta, prompt_tokens, completion_tokens in model.generate(
                prompt,
                stop=stop,
            ):
                if await request.is_disconnected():
                    break
                if stream:
                    yield json.dumps(
                        {
                            "object": "chat.completion.chunk"
                            if chat_mode
                            else "text_completion",
                            "response": response,
                            "tencentcloudresult": tencentcloudresult,
                            "model": model.name,
                            "choices": [
                                {
                                    "delta": {"content": delta},
                                    "index": 0,
                                    "finish_reason": None,
                                }
                                if chat_mode
                                else {
                                    "text": delta,
                                    "index": 0,
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
            # torch_gc()
            requests_num = requests_num - 1
            if await request.is_disconnected():
                print(f"{request.client} Stop Waiting")
                quick_log(
                    request,
                    body,
                    response + "\nStop Waiting. RequestsNum: " + str(requests_num),
                )
                return
            quick_log(
                request,
                body,
                response + "\nFinished. RequestsNum: " + str(requests_num),
            )
            print("response:")
            # print(response)
            if not stream:
                response = filter_response(response)
            response = filter_name(response)
            print("filtered_response:")
            # print(response)
            if stream:
                yield json.dumps(
                    {
                        "object": "chat.completion.chunk"
                        if chat_mode
                        else "text_completion",
                        # "response": response,
                        "model": model.name,
                        "choices": [
                            {
                                "delta": {},
                                "index": 0,
                                "finish_reason": "stop",
                            }
                            if chat_mode
                            else {
                                "text": "",
                                "index": 0,
                                "finish_reason": "stop",
                            }
                        ],
                    }
                )
                yield "[DONE]"
            else:
                yield {
                    "object": "chat.completion" if chat_mode else "text_completion",
                    # "response": response,
                    "model": model.name,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    "choices": [
                        {
                            "message": {
                                "role": Role.Assistant.value,
                                "content": response,
                            },
                            "index": 0,
                            "finish_reason": "stop",
                        }
                        if chat_mode
                        else {
                            "text": response,
                            "index": 0,
                            "finish_reason": "stop",
                        }
                    ],
                }


@router.post("/v1/chat/completions", tags=["Completions"])
@router.post("/chat/completions", tags=["Completions"])
async def chat_completions(body: ChatCompletionBody, request: Request):
    model: TextRWKV = global_var.get(global_var.Model)
    if model is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "model not loaded")

    if body.messages is None or body.messages == []:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "messages not found")

    # print(body.messages[-1].content)
    old_content = body.messages[-1].content
    policy, body.messages[-1].content = filter_request(body.messages[-1].content)
    # print(body.messages[-1].content)
    interface = model.interface
    user = model.user if body.user_name is None else body.user_name
    bot = model.bot if body.assistant_name is None else body.assistant_name

    is_raven = model.rwkv_type == RWKVType.Raven

    completion_text: str = ""
    basic_system: Union[str, None] = None
    if body.presystem:
        if body.messages[0].role == Role.System:
            basic_system = body.messages[0].content

        if basic_system is None:
            completion_text = (
                f"""
The following is a coherent verbose detailed conversation between a girl named {bot} and her friend {user}. \
{bot} is very intelligent, creative and friendly. \
{bot} is unlikely to disagree with {user}, and {bot} doesn't like to ask {user} questions. \
{bot} likes to tell {user} a lot about herself and her opinions. \
{bot} usually gives {user} kind, helpful and informative advices.\n
"""
                if is_raven
                else (
                    f"{user}{interface} hi\n\n{bot}{interface} Hi. "
                    + "I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.\n\n"
                )
            )
        else:
            if not body.messages[0].raw:
                basic_system = (
                    basic_system.replace("\r\n", "\n")
                    .replace("\r", "\n")
                    .replace("\n\n", "\n")
                    .replace("\n", " ")
                    .strip()
                )
            completion_text = (
                (
                    f"The following is a coherent verbose detailed conversation between a girl named {bot} and her friend {user}. "
                    if is_raven
                    else f"{user}{interface} hi\n\n{bot}{interface} Hi. "
                )
                + basic_system.replace("You are", f"{bot} is" if is_raven else "I am")
                .replace("you are", f"{bot} is" if is_raven else "I am")
                .replace("You're", f"{bot} is" if is_raven else "I'm")
                .replace("you're", f"{bot} is" if is_raven else "I'm")
                .replace("You", f"{bot}" if is_raven else "I")
                .replace("you", f"{bot}" if is_raven else "I")
                .replace("Your", f"{bot}'s" if is_raven else "My")
                .replace("your", f"{bot}'s" if is_raven else "my")
                .replace("你", f"{bot}" if is_raven else "我")
                + "\n\n"
            )

    for message in body.messages[(0 if basic_system is None else 1) :]:
        append_message: str = ""
        if message.role == Role.User:
            append_message = f"{user}{interface} " + message.content
        elif message.role == Role.Assistant:
            append_message = f"{bot}{interface} " + message.content
        elif message.role == Role.System:
            append_message = message.content
        if not message.raw:
            append_message = (
                append_message.replace("\r\n", "\n")
                .replace("\r", "\n")
                .replace("\n\n", "\n")
                .strip()
            )
        completion_text += append_message + "\n\n"
    completion_text += f"{bot}{interface}"

    user_code = model.pipeline.decode([model.pipeline.encode(user)[0]])
    bot_code = model.pipeline.decode([model.pipeline.encode(bot)[0]])
    if type(body.stop) == str:
        body.stop = [body.stop, f"\n\n{user_code}", f"\n\n{bot_code}"]
    elif type(body.stop) == list:
        body.stop.append(f"\n\n{user_code}")
        body.stop.append(f"\n\n{bot_code}")
    elif body.stop is None:
        body.stop = default_stop
    if not body.presystem:
        body.stop.append("\n\n")

    if body.stream:
        return EventSourceResponse(
            eval_rwkv(
                model, request, body, completion_text, body.stream, body.stop, True, old_content, policy
            )
        )
    else:
        try:
            return await eval_rwkv(
                model, request, body, completion_text, body.stream, body.stop, True
            ).__anext__()
        except StopAsyncIteration:
            return None


@router.post("/v1/completions", tags=["Completions"])
@router.post("/completions", tags=["Completions"])
async def completions(body: CompletionBody, request: Request):
    model: AbstractRWKV = global_var.get(global_var.Model)
    if model is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "model not loaded")

    if body.prompt is None or body.prompt == "" or body.prompt == []:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "prompt not found")

    if type(body.prompt) == list:
        body.prompt = body.prompt[0]  # TODO: support multiple prompts

    if body.stream:
        return EventSourceResponse(
            eval_rwkv(model, request, body, body.prompt, body.stream, body.stop, False)
        )
    else:
        try:
            return await eval_rwkv(
                model, request, body, body.prompt, body.stream, body.stop, False
            ).__anext__()
        except StopAsyncIteration:
            return None


class EmbeddingsBody(BaseModel):
    input: Union[str, List[str], List[List[int]], None]
    model: Union[str, None] = "rwkv"
    encoding_format: str = None
    fast_mode: bool = False

    model_config = {
        "json_schema_extra": {
            "example": {
                "input": "a big apple",
                "model": "luxi-nlm",
                "encoding_format": None,
                "fast_mode": False,
            }
        }
    }


def embedding_base64(embedding: List[float]) -> str:
    import numpy as np

    return base64.b64encode(np.array(embedding).astype(np.float32)).decode("utf-8")


@router.post("/v1/embeddings", tags=["Embeddings"])
@router.post("/embeddings", tags=["Embeddings"])
@router.post("/v1/engines/text-embedding-ada-002/embeddings", tags=["Embeddings"])
@router.post("/engines/text-embedding-ada-002/embeddings", tags=["Embeddings"])
async def embeddings(body: EmbeddingsBody, request: Request):
    model: AbstractRWKV = global_var.get(global_var.Model)
    if model is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "model not loaded")

    if body.input is None or body.input == "" or body.input == [] or body.input == [[]]:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "input not found")

    global requests_num
    requests_num = requests_num + 1
    quick_log(request, None, "Start Waiting. RequestsNum: " + str(requests_num))
    while completion_lock.locked():
        if await request.is_disconnected():
            requests_num = requests_num - 1
            print(f"{request.client} Stop Waiting (Lock)")
            quick_log(
                request,
                None,
                "Stop Waiting (Lock). RequestsNum: " + str(requests_num),
            )
            return
        await asyncio.sleep(0.1)
    else:
        with completion_lock:
            if await request.is_disconnected():
                requests_num = requests_num - 1
                print(f"{request.client} Stop Waiting (Lock)")
                quick_log(
                    request,
                    None,
                    "Stop Waiting (Lock). RequestsNum: " + str(requests_num),
                )
                return

            base64_format = False
            if body.encoding_format == "base64":
                base64_format = True

            embeddings = []
            prompt_tokens = 0
            if type(body.input) == list:
                if type(body.input[0]) == list:
                    encoding = tiktoken.model.encoding_for_model(
                        "text-embedding-ada-002"
                    )
                    for i in range(len(body.input)):
                        if await request.is_disconnected():
                            break
                        input = encoding.decode(body.input[i])
                        embedding, token_len = model.get_embedding(
                            input, body.fast_mode
                        )
                        prompt_tokens = prompt_tokens + token_len
                        if base64_format:
                            embedding = embedding_base64(embedding)
                        embeddings.append(embedding)
                else:
                    for i in range(len(body.input)):
                        if await request.is_disconnected():
                            break
                        embedding, token_len = model.get_embedding(
                            body.input[i], body.fast_mode
                        )
                        prompt_tokens = prompt_tokens + token_len
                        if base64_format:
                            embedding = embedding_base64(embedding)
                        embeddings.append(embedding)
            else:
                embedding, prompt_tokens = model.get_embedding(
                    body.input, body.fast_mode
                )
                if base64_format:
                    embedding = embedding_base64(embedding)
                embeddings.append(embedding)

            requests_num = requests_num - 1
            if await request.is_disconnected():
                print(f"{request.client} Stop Waiting")
                quick_log(
                    request,
                    None,
                    "Stop Waiting. RequestsNum: " + str(requests_num),
                )
                return
            quick_log(
                request,
                None,
                "Finished. RequestsNum: " + str(requests_num),
            )

            ret_data = [
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": embedding,
                }
                for i, embedding in enumerate(embeddings)
            ]

            return {
                "object": "list",
                "data": ret_data,
                "model": model.name,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "total_tokens": prompt_tokens,
                },
            }
