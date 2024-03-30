# SPDX-FileCopyrightText: 2024-present Alan Meeson <am@carefullycalculated.co.uk>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional

from haystack import DeserializationError, Document, component, default_from_dict, default_to_dict

from sqlite_haystack.document_store import SQLiteDocumentStore


@component
class SQLiteEmbeddingRetriever:
    """
    A component for retrieving documents from an SQLiteDocumentStore using embeddings and vector similarity.
    """

    def __init__(
        self,
        document_store: SQLiteDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = 10,
        num_candidates: Optional[int] = 100,
    ):
        """
        Create the SQLiteEmbeddingRetriever component.

        :param document_store: An instance of SQLiteDocumentStore.
        :param filters: A dictionary with filters to narrow down the search space. Defaults to `None`.
        :param top_k: The maximum number of documents to retrieve. Defaults to `10`.
        :param num_candidates: The number of documents to select with the embedding search, which are then filtered.

        :raises ValueError: If the specified top_k is not > 0.
        :raises ValueError: If the provided document store is not an SQLiteDocumentStore
        """
        if not isinstance(document_store, SQLiteDocumentStore):
            err = "document_store must be an instance of SQLiteDocumentStore"
            raise ValueError(err)

        self._document_store = document_store

        if top_k and top_k <= 0:
            err = f"top_k must be greater than 0. Currently, top_k is {top_k}"
            raise ValueError(err)

        if num_candidates and (num_candidates <= 0):
            err = f"num_candidates must be greater than 0. Currently, num_candidates is {num_candidates}"
            raise ValueError(err)

        self._filters = filters if filters else {}
        self._top_k = top_k
        self._num_candidates = num_candidates

    @component.output_types(documents=List[Document])
    def run(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        num_candidates: Optional[int] = 100,
    ):
        """
        Run the InMemoryEmbeddingRetriever on the given input data.

        :param query_embedding: Embedding of the query.
        :param filters: A dictionary with filters to narrow down the search space.
        :param top_k: The maximum number of documents to return.
        :param num_candidates: The number of documents to select with the embedding search, which are then filtered.
        :return: The retrieved documents.

        :raises ValueError: If the specified DocumentStore is not found or is not an InMemoryDocumentStore instance.
        """
        if filters is None:
            filters = self._filters
        if top_k is None:
            top_k = self._top_k
        if num_candidates is None:
            num_candidates = self._num_candidates

        docs = self._document_store.embedding_retrieval(
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
            num_candidates=num_candidates,
        )

        return {"documents": docs}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        docstore = self._document_store.to_dict()
        return default_to_dict(
            self,
            document_store=docstore,
            filters=self._filters,
            top_k=self._top_k,
            num_candidates=self._num_candidates,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SQLiteEmbeddingRetriever":
        """
        Deserialize this component from a dictionary.
        """
        init_params = data.get("init_parameters", {})
        if "document_store" not in init_params:
            err = "Missing 'document_store' in serialization data"
            raise DeserializationError(err)
        if "type" not in init_params["document_store"]:
            err = "Missing 'type' in document store's serialization data"
            raise DeserializationError(err)
        data["init_parameters"]["document_store"] = SQLiteDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        return default_from_dict(cls, data)
