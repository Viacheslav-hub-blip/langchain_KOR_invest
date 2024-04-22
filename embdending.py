import pandas as pd
from langchain.chains import RetrievalQA, LLMChain
from langchain.vectorstores import DocArrayInMemorySearch, FAISS
from langchain.embeddings import GPT4AllEmbeddings
from langchain.document_loaders import DataFrameLoader
from langchain.chat_models import GigaChat
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

token = 'ZTk3ZjdmYjMtNmMwOC00NGE1LTk0MzktYzA3ZjU4Yzc2YWI3OmY2OGFlMTQ1LTIyNzgtNDIxMC05M2JmLWFhNTFkZjdmYTY1Yw=='

document = [
    "Котировки фьючерсов на золото выросли в ходе американских торгов в пятницу.На COMEX, подразделении Нью-Йоркской "
    "товарной биржи, фьючерсы на золото с поставкой в июне торгуются по цене 2,00 долл. за тройскую унцию, "
    "на момент написания данного комментария поднявшись на 0,37%.Максимумом сессии выступила отметка долл. за "
    "тройскую унцию. На момент написания материала золото нашло поддержку на уровне 2.340,20 долл. и сопротивление — "
    "на 2.433,00 долл.",
    "Investing.com — Цены на золото выросли на азиатских торгах в пятницу, приблизившись к рекордному максимуму после "
    "сообщений об израильских ударах по Ирану, что повысило спрос на «безопасное убежище», особенно в условиях "
    "ухудшения ситуации на Ближнем Востоке.В ходе торгов спотовое золото подорожало до $2417,79 за унцию, "
    "а фьючерс на золото с истекающим сроком действия в июне — до $2433,0 за унцию. Спотовые цены были чуть ниже "
    "рекордного максимума $2430,96 за унцию, достигнутого на прошлой неделе.",
    "Investing.com — Цены на золото выросли в понедельник на азиатских торгах и оказались вблизи рекордных "
    "максимумов, так как спрос на «безопасное убежище» повысился после нападения Ирана на Израиль, хотя укрепление "
    "доллара ограничило значительный рост цен на желтый металл.$2372,62 за унцию в выходные, в то время как июньский "
    "фьючерс на золото остался на уровне $2373,0 за унцию после достижения рекордного максимума $2389,0 за унцию. "
]
model = GigaChat(credentials=token, verify_ssl_certs=False)

# df = pd.DataFrame(document)
# print(df.head())
#
# loader = DataFrameLoader(df, page_content_column='reaction to the event that happened')
# documents = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_documents(documents)

embeddings = GPT4AllEmbeddings(model="./models/ggml-all-MiniLM-L6-v2-f16.bin", n_ctx=512, n_threads=8)
vectorstore = DocArrayInMemorySearch.from_texts(
    document,
    embedding=GPT4AllEmbeddings(model="./models/ggml-all-MiniLM-L6-v2-f16.bin", n_ctx=512, n_threads=8),

)
#db = FAISS.from_documents(texts, embeddings)
retriever = vectorstore.as_retriever()
# vectorstore.save_local('faiss_index')

template = """answer the question based on your knowledge and complete the answer taking into account the context provided below:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

output_parser = StrOutputParser()

setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)

chain = setup_and_retrieval | prompt | model | output_parser

ans = chain.invoke("стоит ли сейчас покупать золото? До какой отметки оно может дойти?")

print(ans)


