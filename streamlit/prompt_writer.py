import requests
import json
import os

#from langchain_upstage import ChatUpstage as Chat
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
#from langchain_community.chat_models import ChatOllama as Chat
from langchain_together import Together as Chat


import streamlit as st


solar = Chat(model="meta-llama/Llama-3-70b-chat-hf", max_tokens=1000)


def get_improved_prompt(prompt):
    # Prompt
    system = """You are an excellent prompt generator. 
    Your goal is to help me write a prompt. 
    For a given prompt, generate better prompt for LLM.

    Response with only one prompt, and prompt text only.          
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Please improve this prompt: \n\n {prompt}"),
        ]
    )

    llm_chain = prompt_template | solar | StrOutputParser()
    return llm_chain.invoke({"prompt": prompt})


def judge_prompt(prompt1, prompt2):
    # Prompt
    system = """
    You are an excellent prompt judger. 
    For given two prompts, you chose one and return the better one.
    Response in the correct json format like {{"best": "PROMPT1"}} or {{"best": "PROMPT2"}}       
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Compare thise two prompts and return the better one: \n\n---\n PROMPT1: \n{prompt1} \n\n---\n PROMPT2: \n{prompt2}",
            ),
        ]
    )

    llm_chain = prompt_template | solar | StrOutputParser()
    return llm_chain.invoke({"prompt1": prompt1, "prompt2": prompt2})


prompt = st.text_area("Enter your prompt to improve")
if st.button("Improve Prompt"):
    for i in range(10):
        st.markdown(f"**Iteration {i+1}:**")
        candidate1 = get_improved_prompt(prompt)
        candidate2 = get_improved_prompt(prompt)

        try:
            judge_result = judge_prompt(candidate1, candidate2)
            judge_result = json.loads(judge_result)
            st.markdown(f"* Prompt1: {candidate1}\n* Prompt2: {candidate2}")

            # lower case the keys

            if judge_result["best"].lower() == "prompt1":
                prompt = candidate1
                st.info("PROMPT1 is better")
            else:
                prompt = candidate2
                st.info("PROMPT2 is better")
        except Exception as e:
            st.error(f"Error: {e}")
    
        st.write(prompt)
        st.divider()


