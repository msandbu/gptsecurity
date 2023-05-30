import sys  
import json
import logging
import os
import msal
import streamlit as st
os.environ["OPENAI_API_KEY"] = "OPENAPIKEY"

output_file = "graph_data.json"
import requests

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
from streamlit_chat import message
from langchain.chains import ConversationChain

config = json.load(open(sys.argv[1]))

app = msal.ConfidentialClientApplication(
    config["client_id"], authority=config["authority"],
    client_credential=config["secret"],
    )

result = None
result = app.acquire_token_silent(config["scope"], account=None)

if not result:
    result = app.acquire_token_for_client(scopes=config["scope"])

if "access_token" in result:
    graph_data = requests.get( 
        config["endpoint"],
        headers={'Authorization': 'Bearer ' + result['access_token']}, ).json()
    with open(output_file, "w") as f:
        json.dump(graph_data, f)    
else:
    print(result.get("error"))
    print(result.get("error_description"))
    print(result.get("correlation_id"))  

JSONReader = download_loader("JSONReader")
loader = JSONReader()
documents = loader.load_data(Path('graph_data.json'))
langchain_documents = [d.to_langchain_format() for d in documents]
llm = OpenAI(temperature=0)
qa_chain = load_qa_chain(llm)


st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.title("GraphGPT : Using LangChain and Streamlit to connect Microsoft Graph to GPT")
 
def load_chain():
    llm = OpenAI(temperature=0)
    chain = ConversationChain(llm=llm)
    return chain

chain = load_chain()
st.header("LangChain Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

def get_text():
    input_text = st.text_input("You: ", "Which Applications have expired?", key="input")
    return input_text

user_input = get_text()
if user_input:
    #output = chain.run(input=user_input)
    output = qa_chain.run(input_documents=langchain_documents, question=user_input)
    

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

answer = qa_chain.run(input_documents=langchain_documents, question=user_input)
print(answer)
