# SPDX-FileCopyrightText: 2024-present Alan Meeson <am@carefullycalculated.co.uk>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock, patch

from haystack.dataclasses import Document

from sqlite_haystack.document_store import SQLiteDocumentStore
from sqlite_haystack.embedding_retriever import SQLiteEmbeddingRetriever


def test_init_default():
    mock_store = Mock(spec=SQLiteDocumentStore)
    retriever = SQLiteEmbeddingRetriever(document_store=mock_store)
    assert retriever._document_store == mock_store
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert retriever._num_candidates == 100


@patch("sqlite_haystack.document_store.SQLiteDocumentStore")
def test_to_dict(_mock_elasticsearch_client):
    document_store = SQLiteDocumentStore(":memory:", embedding_dims=2)
    retriever = SQLiteEmbeddingRetriever(document_store=document_store, num_candidates=50)
    res = retriever.to_dict()

    assert res == {
        "type": "sqlite_haystack.embedding_retriever.SQLiteEmbeddingRetriever",
        "init_parameters": {
            "document_store": {
                "init_parameters": {
                    "database": ":memory:",
                    "use_bm25": True,
                    "embedding_dims": 2,
                    "embedding_similarity_function": "dot_product",
                },
                "type": "sqlite_haystack.document_store.SQLiteDocumentStore",
            },
            "filters": {},
            "top_k": 10,
            "num_candidates": 50,
        },
    }


@patch("sqlite_haystack.document_store.SQLiteDocumentStore")
def test_from_dict(_mock_elasticsearch_client):
    data = {
        "type": "sqlite_haystack.embedding_retriever.SQLiteEmbeddingRetriever",
        "init_parameters": {
            "document_store": {
                "init_parameters": {"database": ":memory:"},
                "type": "sqlite_haystack.document_store.SQLiteDocumentStore",
            },
            "filters": {},
            "top_k": 10,
            "num_candidates": 50,
        },
    }
    retriever = SQLiteEmbeddingRetriever.from_dict(data)
    assert retriever._document_store
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert retriever._num_candidates == 50


def test_run():
    mock_store = Mock(spec=SQLiteDocumentStore)
    mock_store.embedding_retrieval.return_value = [Document(content="Test doc")]
    retriever = SQLiteEmbeddingRetriever(document_store=mock_store)
    res = retriever.run(query_embedding=[0.5, 0.7])
    mock_store.embedding_retrieval.assert_called_once_with(
        query_embedding=[0.5, 0.7],
        filters={},
        top_k=10,
        num_candidates=100,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Test doc"
