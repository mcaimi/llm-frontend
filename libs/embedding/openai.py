#!/usr/bin/env/python

from typing import Callable
from llama_index.embeddings.openai import OpenAIEmbedding


def openai_instance(base_url="http://localhost:11434", model="llama2:7b", api_key=None) -> Callable:
    return OpenAIEmbedding(api_base=base_url, model_name=model, api_key=api_key)
