#!/usr/bin/env python

from .ollama import OllamaEmbeddings
from .openai import OpenAIEmbeddings

class EmbeddingModel(object):
    def __init__(self, config_params: dict) -> None:
        self.config_params = config_params

        if self.config_params.service_type == "openai":
            self.embedding_model = OpenAIEmbeddings(**self.config_params.openai)
        elif self.config_params.service_type == "ollama":
            self.embedding_model = OllamaEmbeddings(**self.config_params.ollama)
        else:
            self.embedding_model = None

    def Model(self):
        return self.embedding_model
