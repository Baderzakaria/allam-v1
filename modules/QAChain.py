from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import logging

class QAChain:
    def __init__(self, model_name, retriever):
        self.model_name = model_name
        self.retriever = retriever

    def create_qa_chain(self):
        llm = Ollama(model=self.model_name)
        prompt_template = """
        ### System:
        You are a respectful and honest assistant. You have to answer the user's 
        questions using only the context provided to you. If you don't know the answer, 
        just say you don't know. Don't try to make up an answer.

        ### Context:
        {context}

        ### User:
        {question}

        ### Response:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self.retriever,
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={'prompt': prompt}
        )

        return qa_chain

    def run(self, query, context_chunks):
        chain = self.create_qa_chain()

        # Use invoke instead of run
        result = chain.invoke({'query': query})

        # Extract the result and source documents
        answer = result['result']
        source_docs = result['source_documents']

        return answer, source_docs
