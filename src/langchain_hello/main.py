import os
from dotenv import load_dotenv

load_dotenv()

information = """
Elon Musk was born on June 28, 1971, in Pretoria, South Africa.
He moved to Canada before eventually settling in the United States.
He co-founded Zip2, a software company, in 1995.
He later co-founded X.com, which became part of PayPal.
He founded SpaceX in 2002, with the goal of reducing space transportation costs.
He joined Tesla in 2004 and became a key figure in the company's development.
"""
from langchain_core.prompts import PromptTemplate

summary_template = """
You are an expert summarizer.

Summarize the following information in exactly three lines.
Include the two three important facts.

Information:
{information}
"""

summary_prompt = PromptTemplate(
    input_variables=["information"],
    template=summary_template
)

from langchain_openai import ChatOpenAI
import httpx

http_client = httpx.Client(
    proxy="socks5://127.0.0.1:10808",
    timeout=60
)

llm = ChatOpenAI(
    model="nvidia/nemotron-3.5-lightning:free",
    api_key=os.getenv("OpenRouter_API_Key"),
    base_url="https://openrouter.ai/api/v1",
    http_client=http_client
)

chain = summary_prompt | llm

response = chain.invoke(
    {
        "information": information
    }
)
print("CONTENT:")
print(response.content)

print("\nUSAGE:")
print(response.usage_metadata)

print("\nMETADATA:")
print(response.response_metadata)

print("\nFULL RESPONSE:")
print(response)