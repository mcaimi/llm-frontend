#!/usr/bin/env/python

from typing import Callable
from langchain_ollama import OllamaEmbeddings


def ollama_instance(base_url="http://localhost:11434", model="llama2:7b") -> Callable:
    return OllamaEmbeddings(base_url=base_url, model=model)
