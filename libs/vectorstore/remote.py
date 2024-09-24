#!/usr/bin/env python

from chromadb import HttpClient
from chromadb.config import Settings


def chroma_client(host: str = "localhost",
                  port: int = 8080,
                  allow_reset: bool = False) -> HttpClient:
    clientSettings: Settings = Settings(allow_reset=allow_reset)

    # create chroma client object
    chromadb_client_http = HttpClient(host=host, port=port,
                                      settings=clientSettings)
    return chromadb_client_http
