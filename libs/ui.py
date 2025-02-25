#!/usr/bin/env python

# import libs
try:
    import gradio as gr
    from .bootup import load_config_parms, get_remote_vectorstore_client
    from langchain_ollama import ChatOllama
    from langchain_openai import ChatOpenAI
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
        self.browser_store = gr.BrowserState([], storage_key=f"_chat_history_{self.config_params.service_type}")
        self.vector_store = get_remote_vectorstore_client(self.config_params)
        if self.config_params.service_type == "openai":
            self.llm = ChatOpenAI(base_url=self.config_params.openai.baseurl, model=self.config_params.openai.model, api_key=self.config_params.openai.apikey)
        elif self.config_params.service_type == "ollama":
            self.llm = ChatOllama(base_url=self.config_params.ollama.baseurl, model=self.config_params.ollama.model)
        else:
            raise Exception(f"Unsupported chat endpoint: {self.config_params.service_type}")

        self.prompt = PromptTemplate.from_template(self.config_params.rag.prompt)
        self.build_chain()

    # read html components
    def html_component(self, path):
        try:
            with open(path) as x:
                return "".join([i.strip() for i in x.readlines()])
        except Exception:
            raise gr.Error(f"Html Component {path} not found", duration=5)

    # build rag chain
    def build_chain(self, rag_switch=False):
        if rag_switch is False:
            def vector_search(message):
                adapter = self.vector_store.Adapter()
                result = adapter.similarity_search_with_score(query=message,
                                                              k=self.config_params.vectorstore.max_objects)

                retrieved_docs = {}
                for res, score in result:
                    retrieved_docs[score] = res
                    print(f"* {score:3f} - [{res.metadata}]")

                # build return object
                context_data = [retrieved_docs[score] for score in sorted(list(retrieved_docs.keys())) if score < self.config_params.vectorstore.score]
                if len(context_data) > self.config_params.vectorstore.max_objects:
                    print(f"Clamping number of results to {self.config_params.vectorstore.max_objects}...")
                    context_data = context_data[:self.config_params.vectorstore.max_objects]

                return "\n\n".join(d.page_content for d in context_data)

            rag_chain = (
                {"context": vector_search, "question": RunnablePassthrough()} | self.prompt | self.llm | StrOutputParser()
            )
        else:
            rag_chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()} | self.prompt | self.llm | StrOutputParser()
            )

        self.rag_chain = rag_chain

    # save chat to local storage
    def save_chat(self, index, chat_contents, chat_store):
        if index is not None:
            chat_store[index] = chat_contents
        else:
            chat_store = chat_store or []
            chat_store.append(chat_contents)
            index = len(chat_store) - 1

        return index, chat_store

    # saved chat title
    def chat_title(self, chat_contents):
        title = ""
        for message in chat_contents:
            if message["role"] == "user":
                if isinstance(message["content"], str):
                    title += message["content"]
                    break
                else:
                    title += "📎 "
        if len(title) > 40:
            title = title[:40] + "..."
        return title or "Saved Conversation"

    # populate chat history
    def load_history(self, chat_store):
        return gr.Dataset(
            samples=[
                [self.chat_title(conv)]
                for conv in chat_store or []
                if conv
            ]
        )

    # chain prediction callback
    def predict(self, message, history):
        msg = " "
        for chunk in self.rag_chain.stream(message.get('text')):
            msg = msg + chunk
            yield msg

    # build interface for a locally hosted model
    def buildUi(self):
        # render interface
        with gr.Blocks(theme=gr.themes.Soft()) as ragInterface:
            gr.HTML(value=self.html_component("assets/header.html"))
            with gr.Row():
                with gr.Column(scale=1):
                    rag_switch = gr.Checkbox(value=False,
                                             visible=True, show_label=True,
                                             label="Bypass RAG",
                                             info="Do not query the Vector DB for relevant embeddings, just go straight to the model.")
                    new_chat_button = gr.Button(
                        "New chat",
                        variant="primary",
                        size="md",
                    )
                    chat_history_dataset = gr.Dataset(
                        components=[gr.Textbox(visible=False)],
                        show_label=False,
                        layout="table",
                        type="index",
                    )

                with gr.Column(scale=5):
                    chatInterface = gr.ChatInterface(self.predict,
                                                     type="messages",
                                                     chatbot=gr.Chatbot(min_height=500, resizeable=True,
                                                                        editable="user", show_copy_button=True, layout="panel",
                                                                        autoscroll=True, type="messages"),
                                                     textbox=gr.MultimodalTextbox(placeholder="Do you need assistance?"),
                                                     multimodal=True,
                                                     theme="soft",
                                                     examples=["What are the advantages of openshift?", "Write a C function that reverses a string"],
                                                     cache_examples=True,
                                                     show_progress="full",
                                                     submit_btn=True,
                                                     editable=True,
                                                     analytics_enabled=False,
                                                     autoscroll=True,
                                                     autofocus=True,
                                                     stop_btn=True,
                                                     flagging_mode="manual",
                                                     flagging_options=["Like", "Spam", "Inappropriate", "Other"],
                                                     )

            # rag bypass switch
            rag_switch.input(fn=self.build_chain, inputs=[rag_switch])

            # chatbot content save callback
            gr.on(triggers=[chatInterface.textbox.submit],
                  fn=self.save_chat,
                  inputs=[chatInterface.conversation_id,
                          chatInterface.chatbot_state,
                          chatInterface.saved_conversations],
                  outputs=[chatInterface.conversation_id, chatInterface.saved_conversations],
                  show_api=False,
                  queue=False,
                  )

            # recall chat from storage callback
            chat_history_dataset.click(
                lambda: [],
                None,
                [chatInterface.chatbot],
                show_api=False,
                queue=False,
                show_progress="hidden",
            ).then(
                chatInterface._load_conversation,
                [chat_history_dataset, chatInterface.saved_conversations],
                [chatInterface.conversation_id, chatInterface.chatbot],
                show_api=False,
                queue=False,
                show_progress="hidden",
            ).then(fn=lambda x: (x, x),
                   inputs=[chatInterface.chatbot],
                   outputs=[chatInterface.chatbot_state, chatInterface.chatbot_value],
                   show_api=False,
                   queue=False
                   )

            # new chat button callback
            new_chat_button.click(
                lambda: (None, []),
                None,
                [chatInterface.conversation_id, chatInterface.chatbot],
                show_api=False,
                queue=False,
            ).then(
                lambda x: x,
                [chatInterface.chatbot],
                [chatInterface.chatbot_state],
                show_api=False,
                queue=False,
            )

            # chat history load/update triggers
            gr.on(
                triggers=[chatInterface.load, chatInterface.saved_conversations.change],
                fn=self.load_history,
                inputs=[chatInterface.saved_conversations],
                outputs=[chat_history_dataset],
                show_api=False,
                queue=False,
            )
        self.web_interface = ragInterface

    # register application in FastAPI
    def registerFastApiEndpoint(self, fastApiApp, path=GRADIO_CUSTOM_PATH):
        fastApiApp = gr.mount_gradio_app(fastApiApp, self.web_interface, path=path)
