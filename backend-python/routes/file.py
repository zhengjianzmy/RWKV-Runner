from zipfile import ZipFile
from fastapi import APIRouter, FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from io import BytesIO
from PyPDF2 import PdfReader
import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity
import openai
import os
import requests
import redis
from _md5 import md5
import json
import docx
from docx import Document
import mammoth
from bs4 import BeautifulSoup
from sse_starlette.sse import EventSourceResponse

# app = FastAPI()

router = APIRouter(include_in_schema=False)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

templates = Jinja2Templates(directory="templates")

# app.mount("/static", StaticFiles(directory="static"), name="static")

db = redis.StrictRedis(host='localhost', port=6379, db=0)


class Chatbot():

    def extract_txt(self, txt):
        with open(txt, "r") as f:
            text = f.read()
        return str(text)

    def extract_pdf(self, pdf):
        print("Parsing paper")
        number_of_pages = len(pdf.pages)
        print(f"Total number of pages: {number_of_pages}")
        paper_text = []
        for i in range(number_of_pages):
            page = pdf.pages[i]
            page_text = []

            def visitor_body(text, cm, tm, fontDict, fontSize):
                x = tm[4]
                y = tm[5]
                # ignore header/footer
                if (y > 50 and y < 720) and (len(text.strip()) > 1):
                    page_text.append({"fontsize": fontSize, "text": text.strip().replace("\x03", ""), "x": x, "y": y})

            _ = page.extract_text(visitor_text=visitor_body)

            blob_font_size = None
            blob_text = ""
            processed_text = []

            for t in page_text:
                if t["fontsize"] == blob_font_size:
                    blob_text += f" {t['text']}"
                    if len(blob_text) >= 200:
                        processed_text.append({"fontsize": blob_font_size, "text": blob_text, "page": i})
                        blob_font_size = None
                        blob_text = ""
                else:
                    if blob_font_size is not None and len(blob_text) >= 1:
                        processed_text.append({"fontsize": blob_font_size, "text": blob_text, "page": i})
                    blob_font_size = t["fontsize"]
                    blob_text = t["text"]
            paper_text += processed_text
        print("Done parsing paper")
        return paper_text

    def extract_text(self, text):
        print("Parsing paper")
        number_of_pages = 1
        print(f"Total number of pages: {number_of_pages}")
        paper_text = []
        for i in range(number_of_pages):
            page = text
            page_text = []

            def visitor_body(text, cm, tm, fontDict, fontSize):
                x = tm[4]
                y = tm[5]
                # ignore header/footer
                if (y > 50 and y < 720) and (len(text.strip()) > 1):
                    page_text.append({"fontsize": fontSize, "text": text.strip().replace("\x03", ""), "x": x, "y": y})

            # _ = page.extract_text(visitor_text=visitor_body)

            blob_font_size = None
            blob_text = ""
            processed_text = []

            for t in page_text:
                if t["fontsize"] == blob_font_size:
                    blob_text += f" {t['text']}"
                    if len(blob_text) >= 200:
                        processed_text.append({"fontsize": blob_font_size, "text": blob_text, "page": i})
                        blob_font_size = None
                        blob_text = ""
                else:
                    if blob_font_size is not None and len(blob_text) >= 1:
                        processed_text.append({"fontsize": blob_font_size, "text": blob_text, "page": i})
                    blob_font_size = t["fontsize"]
                    blob_text = t["text"]
            paper_text += processed_text
        print("Done parsing paper")
        return paper_text

    def extract_text(self, pdf):
        print("Parsing paper")
        paper_text = []

        blob_font_size = None
        blob_text = ""
        processed_text = []

        for t in pdf:
            if t["fontsize"] == blob_font_size:
                blob_text += f" {t['text']}"
                if len(blob_text) >= 200:
                    processed_text.append({"fontsize": blob_font_size, "text": blob_text, "page": t["page"]})
                    blob_font_size = None
                    blob_text = ""
            else:
                if blob_font_size is not None and len(blob_text) >= 1:
                    processed_text.append({"fontsize": blob_font_size, "text": blob_text, "page": t["page"]})
                blob_font_size = t["fontsize"]
                blob_text = t["text"]
        paper_text += processed_text
        print("Done parsing paper")
        return paper_text


    def create_df(self, data):

        if type(data) == list:
            print("Extracting text from pdf")
            print("Creating dataframe")
            filtered_pdf = []
            # print(pdf.pages[0].extract_text())
            for row in data:
                if len(row["text"]) < 30:
                    continue
                filtered_pdf.append(row)
            df = pd.DataFrame(filtered_pdf)
            # remove elements with identical df[text] and df[page] values
            df = df.drop_duplicates(subset=["text", "page"], keep="first")
            # df['length'] = df['text'].apply(lambda x: len(x))
            print("Done creating dataframe")

        elif type(data) == str:
            print("Extracting text from txt")
            print("Creating dataframe")
            # Parse the text and add each paragraph to a column 'text' in a dataframe
            df = pd.DataFrame(data.split("\n"), columns=["text"])

        return df

    def embeddings(self, df):
        print("Calculating embeddings")
        openai.api_key = os.getenv('OPENAI_API_KEY')
        embedding_model = "text-embedding-ada-002"
        embeddings = df.text.apply([lambda x: get_embedding(x, engine=embedding_model)])
        df["embeddings"] = embeddings
        print("Done calculating embeddings")
        return df

    def search(self, df, query, n=3, pprint=True):
        query_embedding = get_embedding(query, engine="text-embedding-ada-002")
        df["similarity"] = df.embeddings.apply(lambda x: cosine_similarity(x, query_embedding))

        results = df.sort_values("similarity", ascending=False, ignore_index=True)
        # make a dictionary of the the first three results with the page number as the key and the text as the value. The page number is a column in the dataframe.
        results = results.head(n)
        sources = []
        for i in range(n):
            # append the page number and the text as a dict to the sources list
            sources.append({"Page " + str(results.iloc[i]["page"]): results.iloc[i]["text"][:150] + "..."})
        return {"results": results, "sources": sources}

    def create_prompt(self, df, user_input):
        print('Creating prompt')
        print(user_input)

        result = self.search(df, user_input, n=3)
        data = result['results']
        sources = result['sources']
        system_role = """You are a AI assistant whose expertise is reading and summarizing scientific papers. You are given a query, 
        a series of text embeddings and the title from a paper in order of their cosine similarity to the query. 
        You must take the given embeddings and return a very detailed summary of the paper in the languange of the query:
        """

        user_input = user_input + """
        Here are the embeddings:

        1.""" + str(data.iloc[0]['text']) + """
        2.""" + str(data.iloc[1]['text']) + """
        3.""" + str(data.iloc[2]['text']) + """
        """

        history = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": str(user_input)}]

        print('Done creating prompt')
        return {'messages': history, 'sources': sources}

    def gpt(self, context, source):
        print('Sending request to OpenAI')
        openai.api_key = os.getenv('OPENAI_API_KEY')
        r = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=context)
        answer = r.choices[0]["message"]["content"]
        print('Done sending request to OpenAI')
        response = {'answer': answer, 'sources': source}
        return response

@router.get("/v1/file", tags=["File"], response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/v1/file/process_pdf", tags=["File"])
async def process_pdf(request: Request):
    print("Processing pdf")
    body = await request.body()
    key = md5(body).hexdigest()
    print(key)

    if db.get(key) is not None:
        print("Already processed pdf")
        return EventSourceResponse(JSONResponse({"key": key}))
    
    chatbot = Chatbot()
    file = body
    try:
        pdf = PdfReader(BytesIO(file))
        paper_text = chatbot.extract_pdf(pdf)
    except Exception as e:
        print(e)
        fileType = 'txt'
        try:
            decode_file = file.decode('UTF-8')
        except Exception as e:
            word_file = BytesIO(file)
            result = mammoth.extract_raw_text(word_file)
            fileType = 'docx'
            # print(result.value)
            decode_file = result.value
        format_txt = decode_file.split('\r\n')
        if len(format_txt) <= 1:
            format_txt = decode_file.split('\n')
        if len(format_txt) <= 1:
            format_txt = decode_file.split('。')
        pdf = []
        page_sum = 3432
        byte_count = 0
        page = 0
        for index, item in enumerate(format_txt):
            byte_count = byte_count + len(item.encode())
            if fileType == 'docx' and byte_count > page_sum:
                page = page + 1
                byte_count = byte_count - page_sum
                print(page)
                print(byte_count)
            if len(item.strip()) <= 1:
                continue
            pdf.append({"fontsize": 10, "text": item.strip().replace("\n", ""), "x": 0, "y": index, "page": page})
        paper_text = chatbot.extract_text(pdf)

    
    df = chatbot.create_df(paper_text)
    print(df)
    df = chatbot.embeddings(df)
    if db.get(key) is None:
        db.set(key, df.to_json())

    print("Done processing pdf")
    return EventSourceResponse(JSONResponse({"key": key}))

@router.post("/v1/file/download_pdf", tags=["File"])
async def download_pdf(url: str):
    chatbot = Chatbot()
    r = requests.get(str(url))
    key = md5(r.content).hexdigest()
    if db.get(key) is not None:
        return JSONResponse({"key": key})
    pdf = PdfReader(BytesIO(r.content))
    paper_text = chatbot.extract_pdf(pdf)
    df = chatbot.create_df(paper_text)
    df = chatbot.embeddings(df)
    if db.get(key) is None:
        db.set(key, df.to_json())
    print("Done processing pdf")
    return JSONResponse({"key": key})

@app.post("/v1/file/reply", tags=["File"])
async def reply(request: Request):
    data = await request.json()
    key = data.get('key')
    print(key)
    # key = '0d1a5394db151b059d3d2ccc9b7159b9'
    query = data.get('query')
    
    chatbot = Chatbot()
    query = str(query)
    df = pd.read_json(BytesIO(db.get(key)))
    print(df.head(5))
    prompt = chatbot.create_prompt(df, query)

    chat = []
    chat.extend(prompt['messages'])

    response = chatbot.gpt(chat, prompt['sources'])
    print(response)
    return JSONResponse(content=response, status_code=200)
