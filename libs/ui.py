#!/usr/bin/env python

# import libs
try:
    import gradio as gr
    from .bootup import load_config_parms, get_remote_vectorstore_client
    from .model.adapter import ChatModel
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
except Exception as e:
    print(f"Caught exception: {e}")
    raise e

# define globals
GRADIO_CUSTOM_PATH = "/rag_ui"

class Ui(object):
    def __init__(self):
        self.web_interface = None
        self.config_params = load_config_parms()
        self.browser_store = gr.BrowserState([], storage_key=f"_chat_history_{self.config_params.service_type}")
        if self.config_params.llm.bypass_rag is False:
            self.vector_store = get_remote_vectorstore_client(self.config_params)

        self.llm_class = ChatModel(self.config_params)
        self.llm = self.llm_class.model()
        if self.llm is None:
            raise Exception(f"Unsupported chat endpoint: {self.config_params.service_type}")

        self.prompt = ChatPromptTemplate([
            ("system", self.config_params.llm.system_prompt),
            ("user", self.config_params.llm.user_prompt)
            ]
        )
        self.build_chain(rag_switch=self.config_params.llm.bypass_rag)

    # read html components
    def html_component(self, path):
        try:
            with open(path) as x:
                return "".join([i.strip() for i in x.readlines()])
        except Exception:
            raise gr.Error(f"Html Component {path} not found", duration=5)

    # rebuild prompt template
    def rebuild_prompt(self, sysprompt: str, userprompt: str, rag_switch: bool):
        if sysprompt == "":
            sysprompt = self.config_params.llm.system_prompt

        if userprompt == "":
            userprompt = self.config_params.llm.user_prompt

        self.prompt = ChatPromptTemplate([
                ("system", sysprompt),
                ("user", userprompt)
            ])

        self.build_chain(rag_switch)

    # build rag chain
    def build_chain(self, rag_switch=False):
        if rag_switch is False:
            self.vector_store = getattr(self, "vector_store", None)
            if self.vector_store is None:
                self.vector_store = get_remote_vectorstore_client(self.config_params)

            def vector_search(message):
                # query the vector store
                nodes_with_score = self.vector_store.Retrieve(query_string=message,
                                                              top_k=self.config_params.vectorstore.max_objects,
                                                              query_mode=self.config_params.vectorstore.query_mode)

                # build return object
                context_data = [valid_node for valid_node in nodes_with_score if valid_node.score < self.config_params.vectorstore.score]
                print(f"Got {len(context_data)} document chunks from the vector database...")
                if len(context_data) > self.config_params.vectorstore.max_objects:
                    print(f"Clamping number of results to {self.config_params.vectorstore.max_objects}...")
                    context_data = context_data[:self.config_params.vectorstore.max_objects]

                return "\n\n".join(d.text for d in context_data)

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

    # count objects in the vector db
    def get_object_count(self, rag_switch) -> str:
        if rag_switch is False:
            self.vector_store = getattr(self, "vector_store", None)
            if self.vector_store is not None:
                return f"Object Count: {self.vector_store.Collection().count()}"
        else:
            return "RAG Bypass Active"

    # chain prediction callback
    def predict(self, message, history):
        msg = " "
        for chunk in self.rag_chain.stream(message.get('text')):
            msg = msg + chunk
            yield msg

    # update llm interface
    def trigger_llm_reload(self, temp, top_k, top_p, num_predict, num_ctx, seed, rag_switch):
        print(f"Refreshing LLM Client T:{temp} top_k: {top_k} top_p: {top_p} num_p: {num_predict} num_ctx: {num_ctx} seed: {seed}")
        self.llm_class.refresh(top_k,
                                 top_p, num_predict,
                                 num_ctx, temp, seed)
        self.llm = self.llm_class.model()
        self.build_chain(rag_switch)

    # build interface for a locally hosted model
    def buildUi(self):
        # render interface
        with gr.Blocks(theme=gr.themes.Soft()) as ragInterface:
            gr.HTML(value=self.html_component("assets/header.html"))
            with gr.Row():
                with gr.Column(scale=1):
                    rag_switch = gr.Checkbox(value=self.config_params.llm.bypass_rag,
                                             visible=True, show_label=True,
                                             label="Bypass RAG",
                                             info="Do not query the Vector DB for relevant embeddings, just go straight to the model.")

                    gr.Textbox(label="AI Backend", value=self.llm_class.model_type, interactive=False)
                    gr.Textbox(label="ChromaDB", value=self.get_object_count, every=gr.Timer(value=20), inputs=[rag_switch], interactive=False)

                    gr.Markdown("### Chat Sessions History")

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
                                                     chatbot=gr.Chatbot(min_height=800, resizeable=True, label="Chat With Assistant",
                                                                        editable="user", show_copy_button=True, layout="panel",
                                                                        avatar_images=("assets/rh_logo.png", "assets/ai_bot.gif"),
                                                                        autoscroll=True, type="messages"),
                                                     textbox=gr.MultimodalTextbox(placeholder="Do you need assistance?"),
                                                     multimodal=True,
                                                     theme="soft",
                                                     examples=["Write a Python script that downloads the index page from google.com using BeautifulSoup.", "Write a C function that reverses a string", "Tell me about RedHat", "How can I load and convert an image from RGB to grayscale using OpenCV"],
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

                with gr.Column(scale=1):
                    sysprompt_value = gr.Textbox(label="System Prompt", value=self.config_params.llm.system_prompt, interactive=True)
                    userprompt_value = gr.Textbox(label="User Prompt", value=self.config_params.llm.user_prompt, interactive=True)
                    prompt_regen_btn = gr.Button("Rebuild Prompt", variant="primary", size="md")
                    llm_temp = gr.Slider(label="Temperature",
                                         minimum=0.0,
                                         maximum=1.0,
                                         step=0.05,
                                         value=self.config_params.llm.temperature,
                                         interactive=True)
                    llm_top_k = gr.Slider(label="top_k",
                                          minimum=0.0,
                                          maximum=100.0,
                                          step=1,
                                          value=self.config_params.llm.top_k,
                                          interactive=True)
                    llm_top_p = gr.Slider(label="top_p",
                                          minimum=0.0,
                                          maximum=1.0,
                                          step=0.05,
                                          value=self.config_params.llm.top_p,
                                          interactive=True)
                    llm_num_predict = gr.Slider(label="Number of Tokens",
                                                minimum=10,
                                                maximum=1024,
                                                step=1,
                                                value=self.config_params.llm.num_predict,
                                                interactive=True)
                    llm_ctx_win = gr.Slider(label="Context Window",
                                            minimum=512,
                                            maximum=8192,
                                            step=1,
                                            value=self.config_params.llm.num_ctx,
                                            interactive=True)
                    llm_seed = gr.Textbox(label="Seed", value=self.config_params.llm.seed, interactive=False)

            # rag bypass switch
            rag_switch.input(fn=self.build_chain, inputs=[rag_switch])

            # rebuild prompt button
            prompt_regen_btn.click(fn=self.rebuild_prompt, inputs=[sysprompt_value, userprompt_value, rag_switch])

            # update llm interface
            gr.on(triggers=[llm_temp.input, llm_top_k.input, llm_top_p.input, llm_num_predict.input, llm_ctx_win.input],
                  fn=self.trigger_llm_reload,
                  inputs=[llm_temp, llm_top_k, llm_top_p, llm_num_predict, llm_ctx_win, llm_seed, rag_switch],
                  show_api=False,
                  queue=False,
                  )

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
