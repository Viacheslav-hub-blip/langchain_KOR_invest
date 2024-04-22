import os
from langchain import PromptTemplate, HuggingFaceHub, LLMChain

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

flan_t5 = HuggingFaceHub(
    repo_id="meta-llama/Meta-Llama-3-8B",
    model_kwargs={"temperature": 1e-10}
)

template = """Question: {question}

Answer: """

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(
    prompt=prompt,
    llm=flan_t5
)

question = "what can the war between Israel and Iran lead to?"

print(llm_chain.run(question))