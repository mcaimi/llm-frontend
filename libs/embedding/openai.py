#!/usr/bin/env/python

from typing import Callable
from langchain_openai import OpenAIEmbeddings


def openai_instance(base_url="http://localhost:11434", model="llama2:7b", api_key=None) -> Callable:
    return OpenAIEmbeddings(openai_api_base=base_url, model=model, api_key=api_key)
