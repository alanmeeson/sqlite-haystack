# SPDX-FileCopyrightText: 2024-present Alan Meeson <am@carefullycalculated.co.uk>
#
# SPDX-License-Identifier: Apache-2.0
import json
import logging
import os
import sqlite3
from typing import Any, Dict, Iterable, List, Literal, Optional, Union

import numpy as np
import sqlite_vss
from haystack import Document, default_from_dict, default_to_dict
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DocumentStore, DuplicatePolicy
from haystack.utils import expit

from sqlite_haystack.filters import _convert_filters_to_where_clause_and_params

logger = logging.getLogger(__name__)

# document scores are essentially unbounded and will be scaled to values between 0 and 1 if scale_score is set to
# True (default). Scaling uses the expit function (inverse of the logit function) after applying a scaling factor
# (e.g., BM25_SCALING_FACTOR for the bm25_retrieval method).
# Larger scaling factor decreases scaled scores.
# For example, an input of 10 is scaled to 0.99 with BM25_SCALING_FACTOR=2
# but to 0.78 with BM25_SCALING_FACTOR=8 (default). The defaults were chosen empirically. Increase the default if most
# unscaled scores are larger than expected (>30) and otherwise would incorrectly all be mapped to scores ~1.
BM25_SCALING_FACTOR = 8
DOT_PRODUCT_SCALING_FACTOR = 100


class SQLiteDocumentStore(DocumentStore):
    """
    Stores data in SQLite, and leverages it's search extensions.
    """

    def __init__(
        self,
        database: Union[str, os.PathLike] = ":memory:",
        *,
        use_bm25: Optional[bool] = True,
        embedding_dims: Optional[int] = None,
        embedding_similarity_function: Literal["dot_product", "cosine"] = "dot_product",
    ):
        """
        Initializes the DocumentStore.

        :param database: The path to the database file to be opened.
        :param use_bm25: If true create a bm25 index.
        :param embedding_dims: dimensions to expect from embedding model. If none, disable embeddings.
        """
        self._db = _create_db(database, use_bm25=use_bm25, embedding_dims=embedding_dims)
        self._database = database
        self._use_bm25 = use_bm25
        self._embedding_dims = embedding_dims
        self.embedding_similarity_function = embedding_similarity_function

    # def __del__(self):
    #    """Close the connection when the object is discarded"""
    #
    #    # TODO: Do I really need to do this?
    #    self._db.close()

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.
        """
        (count,) = self._db.execute("""SELECT COUNT(1) FROM document""").fetchone()
        return count

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Returns the documents that match the filters provided.

        Filters are defined as nested dictionaries that can be of two types:
        - Comparison
        - Logic

        Comparison dictionaries must contain the keys:

        - `field`
        - `operator`
        - `value`

        Logic dictionaries must contain the keys:

        - `operator`
        - `conditions`

        The `conditions` key must be a list of dictionaries, either of type Comparison or Logic.

        The `operator` value in Comparison dictionaries must be one of:

        - `==`
        - `!=`
        - `>`
        - `>=`
        - `<`
        - `<=`
        - `in`
        - `not in`

        The `operator` values in Logic dictionaries must be one of:

        - `NOT`
        - `OR`
        - `AND`


        A simple filter:
        ```python
        filters = {"field": "meta.type", "operator": "==", "value": "article"}
        ```

        A more complex filter:
        ```python
        filters = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.type", "operator": "==", "value": "article"},
                {"field": "meta.date", "operator": ">=", "value": 1420066800},
                {"field": "meta.date", "operator": "<", "value": 1609455600},
                {"field": "meta.rating", "operator": ">=", "value": 3},
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "meta.genre", "operator": "in", "value": ["economy", "politics"]},
                        {"field": "meta.publisher", "operator": "==", "value": "nytimes"},
                    ],
                },
            ],
        }

        :param filters: the filters to apply to the document list.
        :return: a list of Documents that match the given filters.
        """

        if filters:
            query, params = _convert_filters_to_where_clause_and_params(filters)
            combined_query = f"SELECT * FROM document WHERE {query}"  # noqa: S608
            res = self._db.execute(combined_query, params)
        else:
            res = self._db.execute("SELECT * FROM document")

        fields = [f[0] for f in res.description]

        docs = []
        for row in res.fetchall():
            doc_dict = dict(zip(fields, row))
            doc_dict["embedding"] = json.loads(doc_dict["embedding"]) if doc_dict["embedding"] else None
            doc_dict["meta"] = json.loads(doc_dict["meta"]) if doc_dict["meta"] else None
            doc = Document.from_dict(doc_dict)
            docs.append(doc)

        return docs

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Writes (or overwrites) documents into the store.

        :param documents: a list of documents.
        :param policy: documents with the same ID count as duplicates. When duplicates are met,
            the store can:
             - skip: keep the existing document and ignore the new one.
             - overwrite: remove the old document and write the new one.
             - fail: an error is raised
        :raises DuplicateDocumentError: Exception trigger on duplicate document if `policy=DuplicatePolicy.FAIL`
        :return: None
        """
        if (
            not isinstance(documents, Iterable)
            or isinstance(documents, str)
            or any(not isinstance(doc, Document) for doc in documents)
        ):
            err = "Please provide a list of Documents."
            raise ValueError(err)

        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.FAIL

        if policy == DuplicatePolicy.OVERWRITE:
            command = """
                INSERT OR REPLACE INTO document(id, content, dataframe, blob, meta, score, embedding)
                VALUES (?,?,?,?,?,?,?);
            """
        elif policy == DuplicatePolicy.SKIP:
            command = """
                INSERT OR IGNORE INTO document(id, content, dataframe, blob, meta, score, embedding)
                VALUES (?,?,?,?,?,?,?);
            """
        elif policy == DuplicatePolicy.FAIL:
            command = """
                INSERT OR FAIL INTO document(id, content, dataframe, blob, meta, score, embedding)
                VALUES (?,?,?,?,?,?,?);
            """

        doc_dicts = (doc.to_dict(flatten=False) for doc in documents)
        tuples = [
            (
                doc["id"],
                doc["content"],
                doc["dataframe"],
                doc["blob"],
                json.dumps(doc["meta"]),
                doc["score"],
                json.dumps(doc["embedding"]),
            )
            for doc in doc_dicts
        ]

        try:
            cur = self._db.executemany(command, tuples)
            written_documents = cur.rowcount
        except sqlite3.IntegrityError as err:
            raise DuplicateDocumentError() from err

        self._db.commit()

        return written_documents

    def delete_documents(self, object_ids: List[str]) -> None:
        """
        Deletes all documents with a matching document_ids from the document store.
        Fails with `MissingDocumentError` if no document with this id is present in the store.

        :param object_ids: the object_ids to delete
        """

        cur = self._db.cursor()
        values = [[oid] for oid in object_ids]
        cur.executemany("DELETE FROM document WHERE ID = ?", values)
        self._db.commit()

        # TODO: Consider, do I need to handle missing documents?
        # for doc_id in document_ids:  # FIXME
        #    msg = f"ID '{doc_id}' not found, cannot delete it."
        #    raise MissingDocumentError(msg)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this store to a dictionary.
        """
        data = default_to_dict(
            self,
            database=self._database,
            use_bm25=self._use_bm25,
            embedding_dims=self._embedding_dims,
            embedding_similarity_function=self.embedding_similarity_function,
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SQLiteDocumentStore":
        """
        Deserializes the store from a dictionary.
        """
        return default_from_dict(cls, data)

    # below here we have the methods particular to the SQLiteDocument Store
    def bm25_retrieval(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = 10,
        *,
        scale_score: Optional[bool] = False,
    ) -> List[Document]:
        """
        Retrieves documents that are most relevant to the query using BM25 algorithm.

        :param query: The query string.
        :param filters: A dictionary with filters to narrow down the search space.
        :param top_k: The number of top documents to retrieve. Default is 10.
        :param scale_score: Whether to scale the scores of the retrieved documents. Default is False.
        :return: A list of the top_k documents most relevant to the query.
        """
        if not query:
            err = "Query should be a non-empty string"
            raise ValueError(err)

        # Perform the BM25 query
        filter_subclause = ""
        params = []
        if filters:
            filter_query, filter_params = _convert_filters_to_where_clause_and_params(filters)
            filter_subclause = f"WHERE {filter_query}"
            params.extend(list(filter_params))

        limit_subclause = ""
        if top_k:
            limit_subclause = f"LIMIT {top_k!s}"

        query_statement = f"""
            SELECT a.id, a.content, a.dataframe, a.blob, a.meta, b.score, a.embedding
            FROM (
                SELECT * FROM document
                {filter_subclause}
            ) a
            INNER JOIN (
                SELECT id, bm25(document_fts) as score
                FROM document_fts
                WHERE document_fts MATCH(?)
            ) b
            ON a.id = b.id
            ORDER BY b.score
            {limit_subclause}
        """  # noqa: S608

        # Add the query to the parameter set
        params.append(query)

        res = self._db.execute(query_statement, params)
        fields = [f[0] for f in res.description]
        docs = []
        for row in res.fetchall():
            doc_dict = dict(zip(fields, row))
            doc_dict["embedding"] = json.loads(doc_dict["embedding"]) if doc_dict["embedding"] else None
            doc_dict["meta"] = json.loads(doc_dict["meta"]) if doc_dict["meta"] else None
            if scale_score:
                doc_dict["score"] = float(expit(np.asarray(row[1] / 8)))
            doc = Document.from_dict(doc_dict)
            docs.append(doc)

        return docs

    def embedding_retrieval(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = 10,
        num_candidates: Optional[int] = 100,
    ) -> List[Document]:
        """
        Retrieves documents that are most similar to the query embedding using a vector similarity metric.

        :param query_embedding: Embedding of the query.
        :param filters: A dictionary with filters to narrow down the search space.
        :param top_k: The number of top documents to retrieve. Default is 10.
        :param num_candidates: The number of documents to select with the embedding search, which are then filtered.

        :return: A list of the top_k documents most relevant to the query.
        """

        # Perform the Embeddning query
        filter_subclause = ""
        params = []
        if filters:
            filter_query, filter_params = _convert_filters_to_where_clause_and_params(filters)
            filter_subclause = f"WHERE {filter_query}"
            params.extend(list(filter_params))

        limit_subclause = ""
        if top_k:
            limit_subclause = f"LIMIT {top_k!s}"

        embedding_subclause = "vss_search(embedding, vss_search_params(?, ?))"
        params.append(json.dumps(query_embedding))
        params.append(num_candidates)

        query_statement = f"""
            SELECT a.id, a.content, a.dataframe, a.blob, a.meta, b.score, a.embedding
            FROM (
                SELECT rowid, id, content, dataframe, blob, meta, embedding FROM document
                {filter_subclause}
            ) a
            INNER JOIN (
                SELECT rowid, distance as score
                FROM document_vss
                WHERE {embedding_subclause}
            ) b
            ON a.rowid = b.rowid
            ORDER BY b.score
            {limit_subclause}
        """  # noqa: S608

        res = self._db.execute(query_statement, params)

        fields = [f[0] for f in res.description]
        docs = []
        for row in res.fetchall():
            doc_dict = dict(zip(fields, row))
            doc_dict["embedding"] = json.loads(doc_dict["embedding"]) if doc_dict["embedding"] else None
            doc_dict["meta"] = json.loads(doc_dict["meta"]) if doc_dict["meta"] else None
            doc = Document.from_dict(doc_dict)
            docs.append(doc)

        return docs


def _create_db(
    database: Union[str, os.PathLike], *, use_bm25: Optional[bool] = True, embedding_dims: Optional[int] = None
) -> sqlite3.Connection:
    """Opens or Creates the SQLite3 database, and ensures the appropriate tables and indexes are present.

    :param database: The path to the database file to be opened.
    :param use_bm25: If true create a bm25 index.
    :param embedding_dims: dimensions to expect from embedding model. If none, disable embeddings.
    :return: Connection to SQLite database
    """
    db = sqlite3.connect(database)
    if embedding_dims:
        db.enable_load_extension(True)
        sqlite_vss.load(db)

    # Check if documents table exists and create if it doesn't
    db.execute(
        """
            CREATE TABLE IF NOT EXISTS document(
                id TEXT NOT NULL PRIMARY KEY,
                content TEXT,
                dataframe JSON,
                blob BLOB,
                meta JSON,
                score FLOAT,
                embedding JSON
            )
        """
    )

    if use_bm25:
        _create_bm25_index(db)

    if embedding_dims:
        _create_vss_index(db, embedding_dims)

    return db


def _create_bm25_index(db: sqlite3.Connection):
    """Creates the bm25 index table and triggers if they do not already exist."""

    # TODO: look at parameterising the configuration to allow for custom tokenizers, etc.
    # db.execute("""
    #        CREATE VIRTUAL TABLE IF NOT EXISTS document_fts
    #        USING fts5("id", "content", tokenize = 'porter unicode61', content='document', content_rowid='rowid');
    #    """)
    db.execute(
        """
                CREATE VIRTUAL TABLE IF NOT EXISTS document_fts
                USING fts5("id", "content", tokenize = 'trigram', content='document', content_rowid='rowid');
            """
    )

    # Creates triggers that update the index when the documents are added/removed/updated
    db.execute(
        """
            CREATE TRIGGER IF NOT EXISTS document_ai_bm25 AFTER INSERT ON document BEGIN
                INSERT INTO document_fts(rowid, content) VALUES (new.rowid, new.content);
            END;
        """
    )
    db.execute(
        """
        CREATE TRIGGER IF NOT EXISTS document_ad_bm25 AFTER DELETE ON document BEGIN
          INSERT INTO document_fts(document_fts, rowid, content) VALUES('delete', old.rowid, old.content);
        END;
        """
    )
    db.execute(
        """
        CREATE TRIGGER IF NOT EXISTS document_au_bm25 AFTER UPDATE ON document BEGIN
          INSERT INTO document_fts(document_fts, rowid, content) VALUES('delete', old.rowid, old.content);
          INSERT INTO document_fts(rowid, content) VALUES (new.rowid, new.content);
        END;
        """
    )


def _create_vss_index(db: sqlite3.Connection, embedding_dims: int):
    """Creates the vss index table and triggers if they do not already exist."""

    # TODO: look at parameterising the configuration to allow for custom tokenizers, etc.
    db.execute(f"""CREATE VIRTUAL TABLE IF NOT EXISTS document_vss USING vss0(embedding({embedding_dims}));""")

    # Creates triggers that update the index when the documents are added/removed/updated
    db.execute(
        """
            CREATE TRIGGER IF NOT EXISTS document_ai_vss AFTER INSERT ON document BEGIN
                INSERT INTO document_vss(rowid, embedding) VALUES (new.rowid, new.embedding);
            END;
        """
    )
    db.execute(
        """
        CREATE TRIGGER IF NOT EXISTS document_ad_vss AFTER DELETE ON document BEGIN
          DELETE FROM document_vss WHERE rowid = old.rowid;
        END;
        """
    )
    db.execute(
        """
        CREATE TRIGGER IF NOT EXISTS document_au_vss AFTER UPDATE ON document BEGIN
          DELETE FROM document_vss WHERE rowid = old.rowid;
          INSERT INTO document_vss(rowid, embedding) VALUES (new.rowid, new.embedding);
        END;
        """
    )
