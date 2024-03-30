# SPDX-FileCopyrightText: 2024-present Alan Meeson <am@carefullycalculated.co.uk>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock, patch

from haystack.dataclasses import Document

from sqlite_haystack.bm25_retriever import SQLiteBM25Retriever
from sqlite_haystack.document_store import SQLiteDocumentStore


def test_init_default():
    mock_store = Mock(spec=SQLiteDocumentStore)
    retriever = SQLiteBM25Retriever(document_store=mock_store)
    assert retriever._document_store == mock_store
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert not retriever._scale_score


@patch("sqlite_haystack.document_store.SQLiteDocumentStore")
def test_to_dict(_mock_elasticsearch_client):
    document_store = SQLiteDocumentStore(":memory:")
    retriever = SQLiteBM25Retriever(document_store=document_store)
    res = retriever.to_dict()

    assert res == {
        "type": "sqlite_haystack.bm25_retriever.SQLiteBM25Retriever",
        "init_parameters": {
            "document_store": {
                "init_parameters": {
                    "database": ":memory:",
                    "use_bm25": True,
                    "embedding_dims": None,
                    "embedding_similarity_function": "dot_product",
                },
                "type": "sqlite_haystack.document_store.SQLiteDocumentStore",
            },
            "filters": {},
            "top_k": 10,
            "scale_score": False,
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
            "scale_score": True,
        },
    }
    retriever = SQLiteBM25Retriever.from_dict(data)
    assert retriever._document_store
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert retriever._scale_score


def test_run():
    mock_store = Mock(spec=SQLiteDocumentStore)
    mock_store.bm25_retrieval.return_value = [Document(content="Test doc")]
    retriever = SQLiteBM25Retriever(document_store=mock_store)
    res = retriever.run(query="some query")
    mock_store.bm25_retrieval.assert_called_once_with(
        query="some query",
        filters={},
        top_k=10,
        scale_score=False,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Test doc"
