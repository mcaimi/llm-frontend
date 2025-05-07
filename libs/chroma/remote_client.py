#!/usr/bin/env python

from typing import Callable
from chromadb import HttpClient, Collection
from libs.vectorstore.remote import chroma_client
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core import StorageContext
from llama_index.core.schema import NodeWithScore
from typing import Optional

# llamaindex wrapper for a remote chromadb instance
class LlamaIndexChromaRemote(object):
    def __init__(self, host: str = "localhost",
                 port: int = 8080,
                 collection: str = "default",
                 collection_similarity: str = "l2",
                 embedding_function: Callable = None):
        self._client: HttpClient = chroma_client(host=host, port=port)
        if embedding_function is None:
            raise Exception("RemoteChromaClient: embedding_function cannot be None: you must specify an embedding function")
        else:
            self._embed_function = embedding_function
            self._collection: Collection = self._client.get_or_create_collection(collection, metadata={"hnsw:space": collection_similarity})
            self._vector_store: ChromaVectorStore = ChromaVectorStore(chroma_collection=self._collection)
            self._storage_context: StorageContext = StorageContext.from_defaults(vector_store=self._vector_store)

    def Client(self) -> HttpClient:
        return self._client

    def Adapter(self) -> ChromaVectorStore:
        return self._vector_store

    def Collection(self) -> Collection:
        return self._collection

    def Heartbeat(self) -> int:
        return self._client.heartbeat()

    def Retrieve(self, query_string: str, top_k: int, query_mode: str = "default") -> list:
        # embed query
        query_embedding = self._embed_function.embed_query(query_string)
        # query the vector store
        vs_query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=top_k,
                mode=query_mode
                )
        vector_query_result = self.Adapter().query(vs_query)

        # parse resulting nodes and similarity scores
        nodes_with_scores = []
        for index, node in enumerate(vector_query_result.nodes):
            score: Optional[float] = None
            if vector_query_result.similarities is not None:
                score = vector_query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores

    def __str__(self) -> str:
        return f"ChromaDB Client: {self._client.database} - Collection: {self._collection}"
