#!/usr/bin/env python

# import libs
try:
    import os
    import time
    import gradio as gr
    from .bootup import load_config_parms, get_remote_vectorstore_client
    from pathlib import Path
    from langchain_ollama import ChatOllama
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
except Exception as e:
    print(f"Caught exception: {e}")
    raise e

# define globals
GRADIO_CUSTOM_PATH="/rag_ui"

class Ui(object):
    def __init__(self):
        self.web_interface = None
        self.config_params = load_config_parms()
        self.retriever = get_remote_vectorstore_client(self.config_params).Retriever(search_type=self.config_params.vectorstore.search_type, kwargs={"k": self.config_params.vectorstore.max_objects, "score_threshold": self.config_params.vectorstore.score})
        self.llm = ChatOllama(base_url=self.config_params.ollama.baseurl, model=self.config_params.ollama.model)
        self.prompt = PromptTemplate.from_template(self.config_params.rag.prompt)
        self.rag_chain = self.build_chain()

    # read html components
    def html_component(self, path):
        try:
            with open(path) as x:
                return "".join([i.strip() for i in x.readlines()])
        except Exception:
            raise gr.Error(f"Html Component {path} not found", duration=5)

    # build rag chain
    def build_chain(self):
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        return rag_chain

    # chain prediction callback
    def predict(self, message, history):
        msg = " "
        for chunk in self.rag_chain.stream(message):
            msg = msg + chunk
            yield msg

    # build interface for a locally hosted model
    def buildUi(self):
        # render interface
        with gr.Blocks(theme=gr.themes.Soft()) as ragInterface:
            gr.HTML(value=self.html_component("assets/header.html"))
            chatInterface = gr.ChatInterface(
                    self.predict,
                    chatbot=gr.Chatbot(height=600),
                    textbox=gr.Textbox(placeholder="Do you need assistance?", container=False, scale=7),
                    title="RedHat AI",
                    description="This is a demo of an AI enabled application that implements the RAG technique to augment the responses generated by a Large Language Model. The vector DB used is an instance of ChromaDB, while the LLM (Mistral) is served by Ollama. All codebase is pure python and leverages LangChain and FastAPI",
                    theme="soft",
                    examples=["What are the advantages of openshift?", "Write a C function that reverses a string"],
                    cache_examples=True,
                    retry_btn=None,
                    undo_btn="Delete Previous",
                    clear_btn="Clear",
                )

        self.web_interface = ragInterface

    # register application in FastAPI
    def registerFastApiEndpoint(self, fastApiApp, path=GRADIO_CUSTOM_PATH):
        fastApiApp = gr.mount_gradio_app(fastApiApp, self.web_interface, path=path)
