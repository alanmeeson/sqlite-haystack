# SPDX-FileCopyrightText: 2024-present Alan Meeson <am@carefullycalculated.co.uk>
#
# SPDX-License-Identifier: Apache-2.0
import json
import logging
import os
import sqlite3
from typing import Any, Dict, Iterable, List, Optional, Union

from haystack import Document, default_from_dict, default_to_dict
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DocumentStore, DuplicatePolicy

from sqlite_haystack.filters import _convert_filters_to_where_clause_and_params

logger = logging.getLogger(__name__)


class SQLiteDocumentStore(DocumentStore):
    """
    Stores data in SQLite, and leverages it's search extensions.
    """

    def __init__(self, database: Union[str, os.PathLike] = ":memory:"):
        """
        Initializes the DocumentStore.

        :param database: The path to the database file to be opened.
        """
        self._database = database
        self.db = _create_db(database)

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.
        """
        (count,) = self.db.execute("""SELECT COUNT(1) FROM document""").fetchone()
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
            res = self.db.execute(combined_query, params)
        else:
            res = self.db.execute("SELECT * FROM document")

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
            cur = self.db.executemany(command, tuples)
            written_documents = cur.rowcount
        except sqlite3.IntegrityError as err:
            raise DuplicateDocumentError() from err

        self.db.commit()

        return written_documents

    def delete_documents(self, object_ids: List[str]) -> None:
        """
        Deletes all documents with a matching document_ids from the document store.
        Fails with `MissingDocumentError` if no document with this id is present in the store.

        :param object_ids: the object_ids to delete
        """

        cur = self.db.cursor()
        values = [[oid] for oid in object_ids]
        cur.executemany("DELETE FROM document WHERE ID = ?", values)
        self.db.commit()

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
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SQLiteDocumentStore":
        """
        Deserializes the store from a dictionary.
        """
        return default_from_dict(cls, data)


def _create_db(database: Union[str, os.PathLike]) -> sqlite3.Connection:
    """Opens or Creates the SQLite3 database, and ensures the appropriate tables and indexes are present.

    :param database: The path to the database file to be opened.
    :return: Connection to SQLite database
    """

    db = sqlite3.connect(database)

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

    # Check if the retrievers table exists, and create it if it doesn't
    db.execute(
        """
            CREATE TABLE IF NOT EXISTS retrievers(
                id TEXT NOT NULL PRIMARY KEY,
                config JSON
            )
        """
    )

    return db
