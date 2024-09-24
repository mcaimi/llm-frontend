#!/usr/bin/env python

import uuid
from typing import Callable
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from chromadb import HttpClient, Collection
from .vectorstore.remote import chroma_client


class RemoteChromaClient(object):
    def __init__(self, host: str = "localhost",
                 port: int = 8080,
                 collection: str = "default",
                 collection_similarity: str = "l2",
                 embedding_function: Callable = None):
        self._client: HttpClient = chroma_client(host=host, port=port)
        if embedding_function is None:
            raise Exception("RemoteChromaClient: embedding_function cannot be None: you must specify an embedding function")
        else:
            self._collection: Collection = self._client.get_or_create_collection(collection, metadata={"hnsw:space": collection_similarity})
            self._chroma_adapter: Chroma = Chroma(client=self._client, collection_name=collection, embedding_function=embedding_function)

    def Client(self) -> HttpClient:
        return self._client

    def Adapter(self) -> Chroma:
        return self._chroma_adapter

    def Collection(self) -> Collection:
        return self._collection

    def Heartbeat(self) -> int:
        return self._client.heartbeat()

    def Retriever(self, search_type="similarity", kwargs={}) -> VectorStoreRetriever:
        return self._chroma_adapter.as_retriever(search_type=search_type, search_kwargs=kwargs)

    def __str__(self) -> str:
        return f"ChromaDB Client: {self._client.database} - Collection: {self._collection}"
