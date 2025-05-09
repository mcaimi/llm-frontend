#!/usr/bin/env/python

from typing import Callable
from llama_index.embeddings.ollama import OllamaEmbedding


def ollama_instance(base_url="http://localhost:11434", model="llama2:7b") -> Callable:
    return OllamaEmbedding(base_url=base_url, model_name=model)
