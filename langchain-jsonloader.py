import sys 
import json
import logging
import os
os.environ["OPENAI_API_KEY"] = "INSERT OPENAI API KEY HERE"

output_file = "azuredeploy.json"
import requests

# Uses a JSON file locally called azuredeploy.json to inspect

from pathlib import Path
from llama_index import download_loader
from langchain.chains.question_answering import load_qa_chain
from langchain.agents import load_tools
from langchain.agents import initialize_agent 
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.chains.api.prompt import API_RESPONSE_PROMPT
from langchain.chains import APIChain
from langchain.prompts.prompt import PromptTemplate

JSONReader = download_loader("JSONReader")
loader = JSONReader()
documents = loader.load_data(Path('azuredeploy.json'))
langchain_documents = [d.to_langchain_format() for d in documents]
llm = OpenAI(temperature=0)
qa_chain = load_qa_chain(llm)
question="What kind of information is in this JSON file?"
answer = qa_chain.run(input_documents=langchain_documents, question=question)
print(answer)
