#!/usr/bin/env python

import os
import sys
from yaml import safe_load, YAMLError
try:
    from libs.remote_client import RemoteChromaClient
    from libs.utils.console_utils import ANSIColors
    from libs.utils.parameters import Parameters
    from libs.embedding.ollama import ollama_instance
    from libs.embedding.openai import openai_instance
except Exception as e:
    print(f"Caught exception {e}, terminating")
    sys.exit(-1)


def load_config_parms() -> Parameters:
    ttyWriter = ANSIColors()
    config_file = os.getenv("CONFIG_FILE", "parameters.yaml")

    ttyWriter.print_success(text=f"Loading Configuration File {config_file}...")
    try:
        with open(config_file, "r") as f:
            config_parms = safe_load(f)

        parms = Parameters(config_parms)
        return parms
    except YAMLError as e:
        ttyWriter.print_error(text=e)
        raise e
    except Exception as e:
        ttyWriter.print_error(text=e)
        raise e


def get_remote_vectorstore_client(parms: Parameters) -> RemoteChromaClient:
    ttyWriter = ANSIColors()
    if parms.service_type == "ollama":
        ttyWriter.print_warning(f"Running Ollama with model {parms.ollama.model}")
        ttyWriter.print_warning(f"Ollama API URL: {parms.ollama.base_url}")
        embed_func = ollama_instance(base_url=parms.ollama.base_url, model=parms.ollama.model)
    elif parms.service_type == "openai":
        ttyWriter.print_warning(f"Running OpenAI-compatible endpoint with model {parms.openai.model}")
        ttyWriter.print_warning(f"OpenAI API URL: {parms.openai.base_url}")
        ttyWriter.print_warning(f"OpenAI API KEY: {parms.openai.api_key}")
        embed_func = openai_instance(base_url=parms.openai.base_url, model=parms.openai.model, api_key=parms.openai.api_key)
    else:
        ttyWriter.print_error(f"Unsupported embedding endpoint: {parms.service_type}")
        raise Exception(f"{parms.service_type}")

    ttyWriter.print_success("Chroma Client: Initializing Remote Client")
    ttyWriter.print_success(text=f"ChromaDB Instance @ host={parms.chromadb.host}:{parms.chromadb.port}/{parms.chromadb.collection}...")
    try:
        cc = RemoteChromaClient(host=parms.chromadb.host,
                                port=int(parms.chromadb.port),
                                collection=parms.chromadb.collection,
                                collection_similarity=parms.chromadb.collection_similarity,
                                embedding_function=embed_func)
        ttyWriter.print_warning(f"Objects in collection: {cc.Collection().count()}")
        return cc
    except Exception as e:
        ttyWriter.print_error(f"{e}")
