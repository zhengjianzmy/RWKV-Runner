# -*- coding: utf-8 -*-
import os
import sys
import logging
import pymysql
import hashlib
import httpx
import random
import json
import uuid

from datetime import datetime, timedelta
from typing import Union
from jose import JWTError, jwt

from fastapi import APIRouter, FastAPI, Depends, HTTPException, status, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from passlib.context import CryptContext

from fastapi.responses import RedirectResponse, FileResponse
from starlette.requests import Request
from sse_starlette.sse import EventSourceResponse

from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.sms.v20210111 import sms_client, models

router = APIRouter(include_in_schema=False)

class TencentCloudBody(BaseModel):
    phone_number: str

class User(BaseModel):
    username: str
    password: str
    phone_number: str
    email: str

class Category(BaseModel):
    id: str
    input: str
    filtered_input: str
    output: str
    filtered_output: str

class Feedback(BaseModel):
    id: str
    notInteresting: bool
    notTruth: bool
    timeout: bool
    notLogin: bool
    notChat: bool
    bodyHit: bool
    policy: bool
    sex: bool
    notHealthy: bool
    others: bool
    description: str
    phoneNumber: str
    email: str

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def sha256Encrypt(data):
    # 将字符串编码为UTF-8格式，并转换为bytes类型
    data = data.encode("utf-8")
    # 创建SHA256对象
    sha256 = hashlib.sha256()
    # 更新SHA256对象的内容
    sha256.update(data)
    # 获取SHA256对象的摘要信息，返回一个bytes类型的字符串
    digest = sha256.digest()
    # 将bytes类型的字符串转换为十六进制字符串
    encryptedData = digest.hex()
    return encryptedData

# ---- 用pymysql 操作数据库
def get_connection():
    host = 'localhost'
    user = 'root'
    port = 3306
    db = 'luxitech'
    password = os.environ.get("MYSQL_ROOT_PASSWORD")
    conn = pymysql.connect(host=host, port=port, db=db, user=user, password=password)
    return conn

# conn = get_connection()
# 使用 cursor() 方法创建一个 dict 格式的游标对象 cursor
# cursor = conn.cursor(pymysql.cursors.DictCursor)

def get_user_by_username(username: str):
    conn = get_connection()
    # 使用 cursor() 方法创建一个 dict 格式的游标对象 cursor
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    # 使用 execute()  方法执行 SQL 查询
    cursor.execute("SELECT id,username,password,phone_number,email FROM user WHERE username = %s limit 1", username)
    # 使用 fetchone() 方法获取单条数据.
    # 返回为一个根据用户名查询到的字典{'username': 'test', 'password': '9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08'}
    result = cursor.fetchone()
    # 关闭数据库连接
    cursor.close()
    conn.close()
    # print('get_user返回',data)
    if result is not None:
        result = dict(result)
        result = json.dumps(result)
    return result

def get_user_by_phone_number(phone_number: str):
    conn = get_connection()
    # 使用 cursor() 方法创建一个 dict 格式的游标对象 cursor
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    # 使用 execute()  方法执行 SQL 查询
    cursor.execute("SELECT id,username,password,phone_number,email FROM user WHERE phone_number = %s limit 1", phone_number)
    # 使用 fetchone() 方法获取单条数据.
    # 返回为一个根据用户名查询到的字典{'username': 'test', 'password': '9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08'}
    result = cursor.fetchone()
    # 关闭数据库连接
    cursor.close()
    conn.close()
    # print('get_user返回',data)
    if result is not None:
        result = dict(result)
        result = json.dumps(result)
    return result

def insert_register_user(user: User):
    conn = get_connection()
    # 使用 cursor() 方法创建一个 dict 格式的游标对象 cursor
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    # 使用 execute()  方法执行 SQL 查询
    random_uuid = uuid.uuid4()
    now = datetime.now()
    now = now.strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO user (id,username,phone_number,create_time) VALUES(%s,%s,%s,%s)", (random_uuid,user.phone_number,user.phone_number,now))
    # 使用 commit() 方法
    conn.commit()
    # 关闭数据库连接
    cursor.close()
    conn.close()
    # print('insert 成功')
    result = {"uuid": str(random_uuid), "username": user.phone_number, "email": "", "password": ""}
    # print(result)
    result = json.dumps(result)
    # print(result)
    return result

def insert_chat_db(category: Category):
    conn = get_connection()
    # 使用 cursor() 方法创建一个 dict 格式的游标对象 cursor
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    # 使用 execute()  方法执行 SQL 查询
    random_uuid = uuid.uuid4()
    now = datetime.now()
    now = now.strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO category (id,chat_id,input,filtered_input,output,filtered_output,create_time) VALUES(%s,%s,%s,%s,%s,%s,%s)", (category.id,random_uuid,category.input,category.filtered_input,category.output,category.filtered_output,now))
    # 使用 commit() 方法
    conn.commit()
    # 关闭数据库连接
    cursor.close()
    conn.close()
    # print('insert 成功')
    result = {"uuid": str(random_uuid)}
    # print(result)
    result = json.dumps(result)
    # print(result)
    return result

def insert_feedback_db(feedback: Feedback):
    conn = get_connection()
    # # 使用 cursor() 方法创建一个 dict 格式的游标对象 cursor
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    # 使用 execute()  方法执行 SQL 查询
    random_uuid = uuid.uuid4()
    now = datetime.now()
    now = now.strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO feedback (id,feedback_id,is_not_interesting,is_not_truth,is_timeout,is_not_login,is_not_chat,is_body_hit,is_policy,is_sex,is_not_healthy,is_others,description,phone_number,email,create_time) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", (feedback.id,random_uuid,feedback.notInteresting,feedback.notTruth,feedback.timeout,feedback.notLogin,feedback.notChat,feedback.bodyHit,feedback.policy,feedback.sex,feedback.notHealthy,feedback.others,feedback.description,feedback.phoneNumber,feedback.email,now))
    # 使用 commit() 方法
    conn.commit()
    # 关闭数据库连接
    cursor.close()
    conn.close()
    # print('insert 成功')
    result = {"uuid": str(random_uuid)}
    # print(result)
    result = json.dumps(result)
    # print(result)
    return result

def update_user_db(user: User):
    conn = get_connection()
    # 使用 cursor() 方法创建一个 dict 格式的游标对象 cursor
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    # 使用 execute()  方法执行 SQL 查询
    now = datetime.now()
    now = now.strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("UPDATE user SET username=%s,password=%s,email=%s,modify_time=%s WHERE(phone_number=%s)", (user.username,user.password,user.email,now,user.phone_number))
    # 使用 commit() 方法
    conn.commit()
    # 关闭数据库连接
    cursor.close()
    conn.close()
    result = {"username": user.username, "email": user.email, "password": user.password}
    # print(result)
    result = json.dumps(result)
    # print(result)
    return result

def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user.password):
        return False
    return user

class Token(BaseModel):
    access_token: str
    token_type: str
 
def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now() + expires_delta
    else:
        expire = datetime.now() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
    ALGORITHM = "HS256" 
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@router.post("/v1/login/password")
def login_password(user: User):
    return EventSourceResponse(login_by_password(user))

def login_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not verify_password(form_data.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    # 此处应当返回 JWT 令牌
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/v1/login/phone")
def login_phone(user: User):
    return EventSourceResponse(login_by_phone(user))

def login_by_phone(user: User):
    old_user = get_user_by_phone_number(user.phone_number)
    if not old_user:
        result = insert_register_user(user)
        username = user.username
    else:
        old_user = json.loads(old_user)
        result = {"uuid": old_user["id"], "username": old_user["username"], "password": old_user["password"], "email": old_user["email"]}
        result = json.dumps(result)
        username = old_user["username"]
    # print(old_user)
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"username": username}, expires_delta=access_token_expires
    )
    # 此处应当返回 JWT 令牌
    # result = {"access_token": access_token, "token_type": "bearer"}
    # print(result)
    # result = json.dumps(result)
    # print(result)
    yield result

def login_by_password(user: User):
    old_user = get_user_by_phone_number(user.phone_number)
    if not old_user:
        result = {"uuid": "", "username": "", "password": "", "email": ""}
        result = json.dumps(result) 
    else:
        old_user = json.loads(old_user)
        result = {"uuid": old_user["id"], "username": old_user["username"], "password": old_user["password"], "email": old_user["email"]}
        result = json.dumps(result)
    yield result

@router.post("/v1/insert_chat")
def insert_chat(category: Category):
    insert_chat_db(category)
    return {"details": "Insert chat success"}

@router.post("/v1/insert_feedback")
def insert_feedback(feedback: Feedback):
    insert_feedback_db(feedback)
    return {"details": "Insert feedback success"}

@router.post("/v1/update_user")
def update_user(user: User):
    return EventSourceResponse(update_user_info(user))

def update_user_info(user: User):
    result = update_user_db(user)
    yield result

def generate_verification_code(length=6):
    digits = "0123456789"
    code = ""
    for _ in range(length):
        code += random.choice(digits)
    return code

@router.post("/v1/send_verification_code", tags=["Tencentcloud"])
def tencentcloud_sms(data: TencentCloudBody):
    return EventSourceResponse(send_verification_code(data.phone_number))

def send_verification_code(phone_number: str):
    try:
        # 实例化一个认证对象，入参需要传入腾讯云账户 SecretId 和 SecretKey，此处还需注意密钥对的保密
        # 代码泄露可能会导致 SecretId 和 SecretKey 泄露，并威胁账号下所有资源的安全性。以下代码示例仅供参考，建议采用更安全的方式来使用密钥，请参见：https://cloud.tencent.com/document/product/1278/85305
        # 密钥可前往官网控制台 https://console.cloud.tencent.com/cam/capi 进行获取
        cred = credential.Credential(
                os.environ.get("TENCENTCLOUD_SECRET_ID"),
                os.environ.get("TENCENTCLOUD_SECRET_KEY"))
        # 实例化一个http选项，可选的，没有特殊需求可以跳过
        httpProfile = HttpProfile()
        httpProfile.endpoint = "sms.tencentcloudapi.com"

        # 实例化一个client选项，可选的，没有特殊需求可以跳过
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        # 实例化要请求产品的client对象,clientProfile是可选的
        client = sms_client.SmsClient(cred, "ap-nanjing", clientProfile)

        # 实例化一个请求对象,每个接口都会对应一个request对象
        req = models.SendSmsRequest()
        verification_code = generate_verification_code()
        params = {
            "PhoneNumberSet": [ phone_number ],
            "SmsSdkAppId": "1400891390",
            "SignName": "深圳陆兮科技",
            "TemplateId": "2082479",
            "TemplateParamSet": [ verification_code ]
        }
        req.from_json_string(json.dumps(params))

        # 返回的resp是一个SendSmsResponse的实例，与请求对象对应
        resp = client.SendSms(req)
        # 输出json格式的字符串回包
        # print(resp.to_json_string())
        result = {"code": verification_code}
        # print(result)
        result = json.dumps(result)
        # print(result)
        yield result

    except TencentCloudSDKException as err:
        print(err)

