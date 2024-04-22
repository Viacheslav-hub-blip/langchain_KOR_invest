from langchain.prompts import PromptTemplate
from langchain.llms import GigaChat
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import GPT4AllEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from dataset import other_inform, last_month, last_year, dollar

token = 'ZTk3ZjdmYjMtNmMwOC00NGE1LTk0MzktYzA3ZjU4Yzc2YWI3OmY2OGFlMTQ1LTIyNzgtNDIxMC05M2JmLWFhNTFkZjdmYTY1Yw=='

document = other_inform + last_year + last_month + dollar

embeddings = GPT4AllEmbeddings(model="./models/ggml-all-MiniLM-L6-v2-f16.bin", n_ctx=512, n_threads=8)
vectorstore = DocArrayInMemorySearch.from_texts(
    document,
    embedding=GPT4AllEmbeddings(model="./models/ggml-all-MiniLM-L6-v2-f16.bin", n_ctx=512, n_threads=8),

)
retriever = vectorstore.as_retriever()

model = GigaChat(credentials=token, verify_ssl_certs=False)

prompt_1_for_answer = """you have information about the political and economic situation in the world, and you know 
the dollar exchange rate at the moment from the context below.Give an answer about: 1. the current state; 2. the 
stability of the asset; 3. the forecast for the future; 

    {context} 

    Question: {question}    
    """

prompt_2_for_classification = """Given the user message below, you must classify it as either being about BUY,SALE, or WAIT.

    Do not respond with more than one word.

    <message>
    {message}
    </message>   
  
    Classification:"""

prompt_1 = ChatPromptTemplate.from_template(prompt_1_for_answer)

output_parser = StrOutputParser()

setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)

chain = setup_and_retrieval | prompt_1 | model | output_parser

ans = chain.invoke("какой актив стоит покупать в настоящее время")
print(ans)
print()
chain_2 = PromptTemplate.from_template(prompt_2_for_classification) | model | StrOutputParser()
print(chain_2.invoke({"message": ans}))
