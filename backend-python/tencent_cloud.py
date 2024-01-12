# -*- coding: utf-8 -*-
import os
import sys
import logging
import json
import base64

from tencentcloud.common import credential
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
# 导入对应产品模块的client models。
from tencentcloud.tms.v20201229 import tms_client, models

# 导入可选配置类
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
try:
    # 实例化一个认证对象，入参需要传入腾讯云账户secretId，secretKey。
    # 为了保护密钥安全，建议将密钥设置在环境变量中或者配置文件中，请参考本文凭证管理章节。
    # 硬编码密钥到代码中有可能随代码泄露而暴露，有安全隐患，并不推荐。
    # cred = credential.Credential("secretId", "secretKey")
    cred = credential.Credential(
        os.environ.get("TENCENTCLOUD_SECRET_ID"),
        os.environ.get("TENCENTCLOUD_SECRET_KEY"))
    # cred = credential.Credential("SecretId", "SecretKey")

    # 实例化一个http选项，可选的，没有特殊需求可以跳过。
    httpProfile = HttpProfile()
    # 如果需要指定proxy访问接口，可以按照如下方式初始化hp
    # httpProfile = HttpProfile(proxy="http://用户名:密码@代理IP:代理端口")
    httpProfile.scheme = "https"  # 在外网互通的网络环境下支持http协议(默认是https协议),建议使用https协议
    httpProfile.keepAlive = True  # 状态保持，默认是False
    httpProfile.reqMethod = "POST"  # get请求(默认为post请求)
    httpProfile.reqTimeout = 30    # 请求超时时间，单位为秒(默认60秒)
    httpProfile.endpoint = "tms.tencentcloudapi.com"  # 指定接入地域域名(默认就近接入)

    # 实例化一个client选项，可选的，没有特殊需求可以跳过。
    clientProfile = ClientProfile()
    clientProfile.signMethod = "TC3-HMAC-SHA256"  # 指定签名算法
    clientProfile.language = "en-US"  # 指定展示英文（默认为中文）
    clientProfile.httpProfile = httpProfile

    # 实例化要请求产品(以cvm为例)的client对象，clientProfile是可选的。
    client = tms_client.TmsClient(cred, "ap-guangzhou", clientProfile)

    # 打印日志按照如下方式，也可以设置log_format，默认为 '%(asctime)s %(process)d %(filename)s L%(lineno)s %(levelname)s %(message)s'
    # client.set_stream_logger(stream=sys.stdout, level=logging.DEBUG)
    # client.set_file_logger(file_path="/log", level=logging.DEBUG) 日志文件滚动输出，最多10个文件，单个文件最大512MB
    # client.set_default_logger() 去除所有log handler，默认不输出

    # 实例化一个cvm实例信息查询请求对象,每个接口都会对应一个request对象。
    req = models.TextModerationRequest()

    # 填充请求参数,这里request对象的成员变量即对应接口的入参。
    # 您可以通过官网接口文档或跳转到request对象的定义处查看请求参数的定义。
    # respFilter = models.Filter()  # 创建Filter对象, 以zone的维度来查询cvm实例。
    # respFilter.Name = "zone"
    # respFilter.Values = ["ap-shanghai-1", "ap-shanghai-2"]
    # req.Filters = [respFilter]  # Filters 是成员为Filter对象的列表

    # python sdk支持自定义header如 X-TC-TraceId、X-TC-Canary，可以按照如下方式指定，header必须是字典类型的
    # headers = {
    #     "X-TC-TraceId": "ffe0c072-8a5d-4e17-8887-a8a60252abca"
    # }
    # req.headers = headers

    # 要编码的数据
    data = "Hello, World!"
    # 对数据进行编码
    encoded_data = base64.b64encode(data.encode())
    print(encoded_data)
    decoded_data = encoded_data.decode('utf-8')
    print(decoded_data)
    # InputBizType = "1744914766515671040"
    InputBizType = "NLNChat_model_input"
    # OutputBizType = "1744927766656061440"
    OutputBizType = "NLNChat_model_output"
    params = {
        "Content": decoded_data,
        "BizType": InputBizType
    }
    print(params)
    req.from_json_string(json.dumps(params))

    # 通过client对象调用DescribeInstances方法发起请求。注意请求方法名与请求对象是对应的，headers为可选参数。
    # 返回的resp是一个DescribeInstancesResponse类的实例，与请求对象对应。
    resp = client.TextModeration(req)

    # 输出json格式的字符串回包
    print(resp.to_json_string(indent=2))

    # 也可以取出单个值。
    # 您可以通过官网接口文档或跳转到response对象的定义处查看返回字段的定义。
    # print(resp.TotalCount)
except TencentCloudSDKException as err:
    print(err)
