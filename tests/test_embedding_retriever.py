# SPDX-FileCopyrightText: 2024-present Alan Meeson <am@carefullycalculated.co.uk>
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from haystack.dataclasses import Document

from sqlite_haystack.document_store import SQLiteDocumentStore
from sqlite_haystack.embedding_retriever import SQLiteVSSEmbeddingRetriever


# TODO: see if there's a cleaner way of testing an optional package that won't be available on all environments

def test_init_default():
    try:
        import sqlite_vss
        do_test = True
    except ImportError:
        do_test = False

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
            retriever = SQLiteVSSEmbeddingRetriever(document_store=store)


def test_to_dict():
    try:
        import sqlite_vss
        do_test = True
    except ImportError:
        do_test = False

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
    try:
        import sqlite_vss
        expect_not_implemented = False
    except ImportError:
        expect_not_implemented = True

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

    if expect_not_implemented:
        with pytest.raises(NotImplemented):
            retriever = SQLiteVSSEmbeddingRetriever.from_dict(data)
    else:
        retriever = SQLiteVSSEmbeddingRetriever.from_dict(data)
        assert retriever._document_store
        assert retriever._filters == {}
        assert retriever._top_k == 10
        assert retriever._num_candidates == 50
        assert retriever._embedding_dims == 2


def test_run():
    try:
        import sqlite_vss
        do_test = True
    except ImportError:
        do_test = False

    if do_test:
        store = SQLiteDocumentStore(":memory:")
        retriever = SQLiteVSSEmbeddingRetriever(document_store=store, embedding_dims=2)
        store.write_documents([Document(content="Test doc", embedding=[0.5, 0.7])])
        res = retriever.run(query_embedding=[0.5, 0.7])

        assert len(res) == 1
        assert len(res["documents"]) == 1
        assert res["documents"][0].content == "Test doc"

