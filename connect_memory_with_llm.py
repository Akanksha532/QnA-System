# step 1 :setup llm
# step 2 :connect llm with faiss
import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

HF_TOKEN=os.environ.get("HF_TOKEN")
huggingface_repoid='mistralai/Mistral-7B-Instruct-v0.3'
def load_llm(huggingface_repoid):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repoid,
        temperature=0.5,
        task="text-generation",
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"512"}
    )
    return llm

#connect llm  with faiss  and create chain

db_faiss_path="vectorstore\db_faiss"
custom_prompt_template="""
Use the pieces  of information provided in the context to answer the user's question.
If you dont know the answer, just say you dont know, do not try to make up an answer.
Dont provide anything out of the given context

Context:{context}
Question:{question}

start the answer directly. No small talk please"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context","question"])
    return prompt

#load database
embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db=FAISS.load_local(db_faiss_path,embedding_model,allow_dangerous_deserialization=True)

#create qa chain

qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(huggingface_repoid),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(custom_prompt_template)}
)

#invoke chain
user_query=input("Write Your Query:")
response=qa_chain.invoke({'query':user_query})
print("RESULT:",response['result'])
# print('Source Documents:',response['source_documents'])