from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("api_key"))

prompt = ChatPromptTemplate.from_template("What is the capital of {country}?")

chain = prompt | llm

print(chain.invoke({"country": "France"}))
