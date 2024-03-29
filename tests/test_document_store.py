# SPDX-FileCopyrightText: 2024-present Alan Meeson <am@carefullycalculated.co.uk>
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from typing import List
from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.testing.document_store import DocumentStoreBaseTests
from haystack.document_stores.types import DocumentStore, DuplicatePolicy

from sqlite_haystack.document_store import SQLiteDocumentStore


class TestSQLiteDocumentStore(DocumentStoreBaseTests):
    """
    Common test cases will be provided by `DocumentStoreBaseTests` but
    you can add more to this class.
    """

    @pytest.fixture
    def document_store(self) -> SQLiteDocumentStore:
        """
        This is the most basic requirement for the child class: provide
        an instance of this document store so the base class can use it.
        """
        return SQLiteDocumentStore(
            database=":memory:",
            use_bm25=True,
            embedding_dims=384
        )

    def test_write_documents(self, document_store: DocumentStore):
        """
        Test write_documents() fails when trying to write Document with same id
        using DuplicatePolicy.FAIL.
        """
        doc = Document(content="test doc")
        assert document_store.write_documents([doc], policy=DuplicatePolicy.FAIL) == 1
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents(documents=[doc], policy=DuplicatePolicy.FAIL)
        self.assert_documents_are_equal(document_store.filter_documents(), [doc])

    def assert_documents_are_equivalent(self, received: List[Document], expected: List[Document]):
        """
        Assert that two lists of Documents are equivalent; or rather, equal but not necessarily the same order.
        """
        recv = received.sort(key=lambda x: x.id)
        exp = received.sort(key=lambda x: x.id)
        assert recv == exp

    def test_not_operator(self, document_store, filterable_docs):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            filters={
                "operator": "NOT",
                "conditions": [
                    {"field": "meta.number", "operator": "==", "value": 100},
                    {"field": "meta.name", "operator": "==", "value": "name_0"},
                ],
            }
        )

        self.assert_documents_are_equivalent(
            result, [d for d in filterable_docs if not (d.meta.get("number") == 100 and d.meta.get("name") == "name_0")]
        )
