from langchain_community.chat_models import GigaChat
from langchain_community.llms import GigaChat as LLM
from langchain.agents import (
    load_tools,
    initialize_agent,
    AgentType,
    Tool,
    AgentExecutor,

)
from langchain.agents import create_pandas_dataframe_agent


import pandas as pd

token = 'ZTk3ZjdmYjMtNmMwOC00NGE1LTk0MzktYzA3ZjU4Yzc2YWI3OmY2OGFlMTQ1LTIyNzgtNDIxMC05M2JmLWFhNTFkZjdmYTY1Yw=='

model = GigaChat(credentials=token, verify_ssl_certs=False)
llm = LLM(credentials=token, verify_ssl_certs=False)
text = 'What would be a good company name for a company that makes colorful socks?'
# print(llm.invoke(text))

tools = load_tools(["llm-math"], llm=llm)
# agent = initialize_agent(
#     tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
# )

# agent.run(
#     "what is the median salary of senior data scientists in 2023? what is the figure given there is a 10% increment?"
# )

df = pd.read_excel("ds_salaries.xlsx")

# agent = create_csv_agent(llm=model, path="ds_salaries.csv", verbose=True, handle_parsing_errors=True)
# print(agent)
# agent.run("how many rows are there?")
