#!/usr/bin/env python

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

class ChatModel(object):
    def __init__(self, config_params: dict) -> None:
        self.config_params = config_params

        if self.config_params.service_type == "openai":
            self.llm = ChatOpenAI(base_url = self.config_params.openai.base_url,
                                  model = self.config_params.openai.model,
                                  api_key = self.config.openai.api_key,
                                  temperature = self.config_params.llm.temperature,
                                  max_tokens = self.config_params.llm.num_predict)
        elif self.config_params.service_type == "ollama":
            self.llm = ChatOllama(base_url = self.config_params.ollama.base_url,
                                  model = self.config_params.ollama.model,
                                  top_k = self.config_params.llm.top_k,
                                  top_p = self.config_params.llm.top_p,
                                  num_predict = self.config_params.llm.num_predict,
                                  num_ctx = self.config_params.llm.num_ctx,
                                  temperature = self.config_params.llm.temperature,
                                  seed = self.config_params.llm.seed)
        else:
            self.llm = None

    def refresh(self, top_k, top_p, num_predict, num_ctx, temperature, seed):
        if self.config_params.service_type == "openai":
            self.llm = ChatOpenAI(base_url = self.config_params.openai.base_url,
                                  model = self.config_params.openai.model,
                                  api_key = self.config.openai.api_key,
                                  temperature = temperature,
                                  max_tokens = num_predict)
        elif self.config_params.service_type == "ollama":
            self.llm = ChatOllama(base_url = self.config_params.ollama.base_url,
                                  model = self.config_params.ollama.model,
                                  top_k = top_k,
                                  top_p = top_p,
                                  num_predict = num_predict,
                                  num_ctx = num_ctx,
                                  temperature = temperature,
                                  seed = seed)
        else:
            self.llm = None

    def model(self):
        return self.llm

    def model_type(self) -> str:
        if self.config_params.service_type == "ollama":
            return f"Service: {self.config_params.service_type} ({self.config_params.ollama.base_url}) - Model: {self.config_params.ollama.model}"
        elif self.config_params.service_type == "openai":
            return f"Service: {self.config_params.service_type} ({self.config_params.openai.base_url}) - Model: {self.config_params.openai.model}"
        else:
            return f"Service: {self.config_params.service_type} unsupported"
