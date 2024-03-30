# SPDX-FileCopyrightText: 2024-present Alan Meeson <am@carefullycalculated.co.uk>
#
# SPDX-License-Identifier: Apache-2.0
import importlib.util

import pytest
from haystack.dataclasses import Document

from sqlite_haystack.document_store import SQLiteDocumentStore
from sqlite_haystack.embedding_retriever import SQLiteVSSEmbeddingRetriever

# TODO: see if there's a cleaner way of testing an optional package that won't be available on all environments


def test_init_default():
    do_test = importlib.util.find_spec("sqlite_vss") is not None

    store = SQLiteDocumentStore(":memory:")
    if do_test:
        # If we can import sqlite-vss we test the component
        retriever = SQLiteVSSEmbeddingRetriever(document_store=store)
        assert retriever._document_store == store
        assert retriever._filters == {}
        assert retriever._top_k == 10
        assert retriever._num_candidates == 100
    else:
        # If we can't then we test the "This optional component is disabled" logic
        with pytest.raises(NotImplemented):
            _ = SQLiteVSSEmbeddingRetriever(document_store=store)


def test_to_dict():
    do_test = importlib.util.find_spec("sqlite_vss") is not None

    if do_test:
        document_store = SQLiteDocumentStore(":memory:")
        retriever = SQLiteVSSEmbeddingRetriever(document_store=document_store, num_candidates=50, embedding_dims=2)
        res = retriever.to_dict()

        assert res == {
            "type": "sqlite_haystack.embedding_retriever.SQLiteVSSEmbeddingRetriever",
            "init_parameters": {
                "document_store": {
                    "init_parameters": {
                        "database": ":memory:",
                    },
                    "type": "sqlite_haystack.document_store.SQLiteDocumentStore",
                },
                "filters": {},
                "top_k": 10,
                "num_candidates": 50,
                "embedding_dims": 2,
            },
        }


def test_from_dict():
    do_test = importlib.util.find_spec("sqlite_vss") is not None

    data = {
        "type": "sqlite_haystack.embedding_retriever.SQLiteVSSEmbeddingRetriever",
        "init_parameters": {
            "document_store": {
                "init_parameters": {"database": ":memory:"},
                "type": "sqlite_haystack.document_store.SQLiteDocumentStore",
            },
            "filters": {},
            "top_k": 10,
            "num_candidates": 50,
            "embedding_dims": 2,
        },
    }

    if do_test:
        retriever = SQLiteVSSEmbeddingRetriever.from_dict(data)
        assert retriever._document_store
        assert retriever._filters == {}
        assert retriever._top_k == 10
        assert retriever._num_candidates == 50
        assert retriever._embedding_dims == 2
    else:
        with pytest.raises(NotImplemented):
            _ = SQLiteVSSEmbeddingRetriever.from_dict(data)


def test_run():
    do_test = importlib.util.find_spec("sqlite_vss") is not None

    if do_test:
        store = SQLiteDocumentStore(":memory:")
        retriever = SQLiteVSSEmbeddingRetriever(document_store=store, embedding_dims=2)
        store.write_documents([Document(content="Test doc", embedding=[0.5, 0.7])])
        res = retriever.run(query_embedding=[0.5, 0.7])

        assert len(res) == 1
        assert len(res["documents"]) == 1
        assert res["documents"][0].content == "Test doc"
