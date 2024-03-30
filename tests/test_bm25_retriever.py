# SPDX-FileCopyrightText: 2024-present Alan Meeson <am@carefullycalculated.co.uk>
#
# SPDX-License-Identifier: Apache-2.0
import sqlite3
from unittest.mock import Mock, patch

from haystack.dataclasses import Document

from sqlite_haystack.bm25_retriever import SQLiteBM25Retriever
from sqlite_haystack.document_store import SQLiteDocumentStore


def test_init_default():
    mock_store = Mock(spec=SQLiteDocumentStore)
    mock_store.db = Mock(spec=sqlite3.Connection)
    retriever = SQLiteBM25Retriever(document_store=mock_store)
    assert retriever._document_store == mock_store
    assert retriever._filters == {}
    assert retriever._top_k == 10


@patch("sqlite_haystack.document_store.SQLiteDocumentStore")
def test_to_dict(_mock_elasticsearch_client):
    document_store = SQLiteDocumentStore(":memory:")
    retriever = SQLiteBM25Retriever(document_store=document_store)
    res = retriever.to_dict()

    assert res == {
        "type": "sqlite_haystack.bm25_retriever.SQLiteBM25Retriever",
        "init_parameters": {
            "document_store": {
                "init_parameters": {"database": ":memory:"},
                "type": "sqlite_haystack.document_store.SQLiteDocumentStore",
            },
            "filters": {},
            "top_k": 10,
            "tokenizer": "trigram",
            "snippet": False,
            "snippet_prefix": "<b>",
            "snippet_suffix": "</b>",
            "snippet_max_tokens": 64,
            "highlight": False,
            "highlight_prefix": "<b>",
            "highlight_suffix": "</b>",
        },
    }


@patch("sqlite_haystack.document_store.SQLiteDocumentStore")
def test_from_dict(_mock_elasticsearch_client):

    data = {
        "type": "sqlite_haystack.bm25_retriever.SQLiteBM25Retriever",
        "init_parameters": {
            "document_store": {
                "init_parameters": {"database": ":memory:"},
                "type": "sqlite_haystack.document_store.SQLiteDocumentStore",
            },
            "filters": {},
            "top_k": 10,
            "tokenizer": "trigram",
            "snippet": False,
            "snippet_prefix": "<b>",
            "snippet_suffix": "</b>",
            "snippet_max_tokens": 64,
            "highlight": False,
            "highlight_prefix": "<b>",
            "highlight_suffix": "</b>",
        },
    }
    retriever = SQLiteBM25Retriever.from_dict(data)
    assert retriever._document_store
    assert retriever._filters == {}
    assert retriever._top_k == 10


def test_run():
    store = SQLiteDocumentStore(":memory:")
    retriever = SQLiteBM25Retriever(document_store=store)
    store.write_documents([Document(content="Test doc expecting some query")])
    res = retriever.run(query="some query")
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Test doc expecting some query"
