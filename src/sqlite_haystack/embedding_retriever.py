# SPDX-FileCopyrightText: 2024-present Alan Meeson <am@carefullycalculated.co.uk>
#
# SPDX-License-Identifier: Apache-2.0
import json
import sqlite3
from typing import Any, Dict, List, Optional

from haystack import DeserializationError, Document, component, default_from_dict, default_to_dict

from sqlite_haystack.document_store import SQLiteDocumentStore
from sqlite_haystack.filters import _convert_filters_to_where_clause_and_params

try:
    import sqlite_vss

    @component
    class SQLiteVSSEmbeddingRetriever:
        """
        A component for retrieving documents from an SQLiteDocumentStore using embeddings and vector similarity.
        """

        NAME = "sqlite_haystack.embedding_retriever.SQLiteVSSEmbeddingRetriever"

        def __init__(
            self,
            document_store: SQLiteDocumentStore,
            filters: Optional[Dict[str, Any]] = None,
            top_k: Optional[int] = 10,
            num_candidates: int = 100,
            embedding_dims: int = 384,
        ):
            """
            Create the SQLiteEmbeddingRetriever component.

            :param document_store: An instance of SQLiteDocumentStore.
            :param filters: A dictionary with filters to narrow down the search space. Defaults to `None`.
            :param top_k: The maximum number of documents to retrieve. Defaults to `10`.
            :param num_candidates: The number of documents to select with the embedding search, which are then filtered.
            :param embedding_dims: The size of the embedding vector; used when creating the index.

            :raises ValueError: If the specified top_k is not > 0.
            :raises ValueError: If the provided document store is not an SQLiteDocumentStore
            """
            if not isinstance(document_store, SQLiteDocumentStore):
                err = "document_store must be an instance of SQLiteDocumentStore"
                raise ValueError(err)

            self._document_store = document_store

            if top_k and (top_k <= 0):
                err = f"top_k must be greater than 0. Currently, top_k is {top_k}"
                raise ValueError(err)

            if num_candidates <= 0:
                err = f"num_candidates must be greater than 0. Currently, num_candidates is {num_candidates}"
                raise ValueError(err)

            self._filters = filters if filters else {}
            self._top_k = top_k
            self._num_candidates = num_candidates
            self._embedding_dims = embedding_dims

            self._create_vss_index()

        @component.output_types(documents=List[Document])
        def run(
            self,
            query_embedding: List[float],
            filters: Optional[Dict[str, Any]] = None,
            top_k: Optional[int] = None,
            num_candidates: Optional[int] = None,
        ):
            """
            Run the SQLiteEmbeddingRetriever on the given input data.

            :param query_embedding: Embedding of the query.
            :param filters: A dictionary with filters to narrow down the search space.
            :param top_k: The maximum number of documents to return.
            :param num_candidates: The number of documents to select with the embedding search, which are then filtered.
            :return: The retrieved documents.
            """
            filters = filters if filters else self._filters
            top_k = top_k if top_k else self._top_k
            num_candidates = num_candidates if num_candidates else self._num_candidates

            # Perform the Embedding query
            filter_subclause = ""
            params = []
            if filters:
                filter_query, filter_params = _convert_filters_to_where_clause_and_params(filters)
                filter_subclause = f"WHERE {filter_query}"
                params.extend(list(filter_params))

            embedding_subclause = "vss_search(embedding, vss_search_params(?, ?))"
            params.append(json.dumps(query_embedding))
            params.append(num_candidates)

            limit_subclause = ""
            if top_k:
                limit_subclause = "LIMIT ?"
                params.append(top_k)

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

            cursor = self._document_store.db.cursor()
            res = cursor.execute(query_statement, params)

            fields = [f[0] for f in res.description]
            docs = []
            for row in res.fetchall():
                doc_dict = dict(zip(fields, row))
                doc_dict["embedding"] = json.loads(doc_dict["embedding"]) if doc_dict["embedding"] else None
                doc_dict["meta"] = json.loads(doc_dict["meta"]) if doc_dict["meta"] else None
                doc = Document.from_dict(doc_dict)
                docs.append(doc)

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
                embedding_dims=self._embedding_dims,
            )

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> "SQLiteVSSEmbeddingRetriever":
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

        def _create_vss_index(self):
            """Creates the vss index table and triggers if they do not already exist."""

            self._document_store.db.enable_load_extension(True)
            sqlite_vss.load(self._document_store.db)

            cursor = self._document_store.db.cursor()
            cursor.execute("BEGIN TRANSACTION;")
            try:
                # TODO: Make this bit do something
                # If a config entry exists in the datasource, grab it
                # res = cursor.execute("SELECT config FROM retrievers WHERE id = ?", [self.NAME]).fetchone()
                # if res:
                #    config = json.loads(res[0])

                cursor.execute(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS document_vss USING vss0(embedding({self._embedding_dims}));"
                )

                # Creates triggers that update the index when the documents are added/removed/updated
                cursor.execute(
                    """
                        CREATE TRIGGER IF NOT EXISTS document_ai_vss AFTER INSERT ON document BEGIN
                            INSERT INTO document_vss(rowid, embedding) VALUES (new.rowid, new.embedding);
                        END;
                    """
                )

                cursor.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS document_ad_vss AFTER DELETE ON document BEGIN
                      DELETE FROM document_vss WHERE rowid = old.rowid;
                    END;
                    """
                )

                cursor.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS document_au_vss AFTER UPDATE ON document BEGIN
                      DELETE FROM document_vss WHERE rowid = old.rowid;
                      INSERT INTO document_vss(rowid, embedding) VALUES (new.rowid, new.embedding);
                    END;
                    """
                )

                # Add the retrievers table
                config = self.to_dict()
                config = config["init_parameters"]
                if "document_store" in config:
                    del config["document_store"]
                config_json = json.dumps(config)

                cursor.execute(
                    """INSERT OR REPLACE INTO retrievers(id, config) VALUES(?, ?);""", [self.NAME, config_json]
                )

            except sqlite3.Error as err:
                self._document_store.db.rollback()
                raise err
            finally:
                self._document_store.db.commit()

except ImportError:

    # If we failed to import sqlite-vss it's probably because the user is on a platform that doesn't yet support it.
    # So, we define the class as not implemented.  The 'ignore' is because mypy doesn't understand this pattern.
    class SQLiteVSSEmbeddingRetriever:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise NotImplementedError

        @classmethod
        def from_dict(cls, *args, **kwargs):
            raise NotImplementedError
