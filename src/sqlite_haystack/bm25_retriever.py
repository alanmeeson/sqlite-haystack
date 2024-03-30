# SPDX-FileCopyrightText: 2024-present Alan Meeson <am@carefullycalculated.co.uk>
#
# SPDX-License-Identifier: Apache-2.0
import json
import sqlite3
from typing import Any, Dict, List, Optional

from haystack import DeserializationError, Document, component, default_from_dict, default_to_dict

from sqlite_haystack.document_store import SQLiteDocumentStore
from sqlite_haystack.filters import _convert_filters_to_where_clause_and_params


@component
class SQLiteBM25Retriever:
    """
    A component for retrieving documents from an SQLiteDocumentStore using the BM25 algorithm.
    """

    NAME = "sqlite_haystack.bm25_retriever.SQLiteBM25Retriever"

    def __init__(
        self,
        document_store: SQLiteDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = 10,
        *,
        tokenizer: str = "trigram",
        snippet: bool = False,
        snippet_prefix: str = "<b>",
        snippet_suffix: str = "</b>",
        snippet_max_tokens: int = 64,
        highlight: bool = False,
        highlight_prefix: str = "<b>",
        highlight_suffix: str = "</b>",
    ):
        """
        Create an SQLiteBM25Retriever component. Usually you pass some basic configuration
        parameters to the constructor.

        :param document_store: A Document Store object used to retrieve documents
        :param filters: A dictionary with filters to narrow down the search space (default is None).
        :param top_k: The maximum number of documents to retrieve (default is 10).
        :param tokenizer: specify the tokenizer to use when performing searches.
            See: https://sqlite.org/fts5.html#tokenizers
        :param snippet: if True adds `snippet` metadata field to the results containing a snippet of the content which
            contained the greatest number of matching terms possible.  Matching terms are wrapped with the values of
            `snippet_prefix` and `snippet_suffix`.
        :param snippet_prefix: text to place before matching sections
        :param snippet_suffix: text to place after matching sections
        :param snippet_max_tokens: The maximum number of tokens to include in the snippet.  Max allowed is 64, min is 1.
        :param highlight: if True adds `highlight` metadata field to the results containing a copy of the context with
            the matching sections wrapped with the values of `highlight_prefix` and `highlight_suffix`
        :param highlight_prefix: text to place before highlighted sections
        :param highlight_suffix: text to place after highlighted sections
        :raises ValueError: If the specified top_k is not > 0.
        :raises ValueError: If the provided document store is not an SQLiteDocumentStore
        """
        if not isinstance(document_store, SQLiteDocumentStore):
            err = "document_store must be an instance of SQLiteDocumentStore"
            raise ValueError(err)

        self._document_store = document_store

        if top_k and top_k <= 0:
            err = f"top_k must be greater than 0. Currently, the top_k is {top_k}"
            raise ValueError(err)

        # Maximum number of tokens supported by the sqlite3 fts snippet function is 64. ALL_CAPS_TO_INDICATE_CONSTANT.
        MAX_SUPPORTED_TOKENS = 64  # noqa: N806
        if snippet_max_tokens <= 0 or snippet_max_tokens > MAX_SUPPORTED_TOKENS:
            err = f"snippet_max_tokens must be >0 and <= {MAX_SUPPORTED_TOKENS}, but {snippet_max_tokens} was provided."
            raise ValueError(err)

        self._filters = filters if filters else {}
        self._top_k = top_k
        self._tokenizer = tokenizer
        self._snippet = snippet
        self._snippet_prefix = snippet_prefix
        self._snippet_suffix = snippet_suffix
        self._snippet_max_tokens = snippet_max_tokens
        self._highlight = highlight
        self._highlight_prefix = highlight_prefix
        self._highlight_suffix = highlight_suffix

        self._create_bm25_index()

    @component.output_types(documents=List[Document])
    def run(self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = 10):
        """
        Run the SQLiteBM25Retriever on the given input data.

        :param query: The query string for the Retriever.
        :param filters: A dictionary with filters to narrow down the search space.
        :param top_k: The maximum number of documents to return.
        :return: The retrieved documents.

        :raises ValueError: If the specified DocumentStore is not found or is not a InMemoryDocumentStore instance.
        """
        filters = filters if filters else self._filters
        top_k = top_k if top_k else self._top_k

        if top_k and top_k <= 0:
            err = f"top_k must be greater than 0. Currently, the top_k is {top_k}"
            raise ValueError(err)

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

        main_selection_clause = "a.id, a.content, a.dataframe, a.blob, a.meta, b.score, a.embedding"
        rank_section_clause = "id, rank as score"
        if self._snippet:
            main_selection_clause += ", b._snippet"
            rank_section_clause += ", snippet(document_fts, 1, ?, ?, '...', ?) as _snippet"
            params.append(self._snippet_prefix)
            params.append(self._snippet_suffix)
            params.append(self._snippet_max_tokens)

        if self._highlight:
            main_selection_clause += ", b._highlight"
            rank_section_clause += ", highlight(document_fts, 1, ?, ?) as _highlight"
            params.append(self._highlight_prefix)
            params.append(self._highlight_suffix)

        # Add the query to the parameter set
        params.append(query)

        limit_subclause = ""
        if top_k:
            limit_subclause = "LIMIT ?"
            params.append(top_k)

        query_statement = f"""
            SELECT {main_selection_clause}
            FROM (
                SELECT * FROM document
                {filter_subclause}
            ) a
            INNER JOIN (
                SELECT {rank_section_clause}
                FROM document_fts
                WHERE document_fts MATCH(?)
            ) b
            ON a.id = b.id
            ORDER BY b.score
            {limit_subclause}
        """  # noqa: S608

        cursor = self._document_store.db.cursor()
        res = cursor.execute(query_statement, params)

        fields = [f[0] for f in res.description]
        docs = []

        for row in res.fetchall():
            # map the standard fields into the dict.
            doc_dict = dict(zip(fields, row))

            # convert the embeddings from json to a list
            doc_dict["embedding"] = json.loads(doc_dict["embedding"]) if doc_dict["embedding"] else None

            # convert the metadata from json to a dict
            doc_dict["meta"] = json.loads(doc_dict["meta"]) if doc_dict["meta"] else {}

            if "_snippet" in doc_dict:
                doc_dict["meta"]["snippet"] = doc_dict["_snippet"]
                del doc_dict["_snippet"]

            if "_highlight" in doc_dict:
                doc_dict["meta"]["highlight"] = doc_dict["_highlight"]
                del doc_dict["_highlight"]

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
            tokenizer=self._tokenizer,
            snippet=self._snippet,
            snippet_prefix=self._snippet_prefix,
            snippet_suffix=self._snippet_suffix,
            snippet_max_tokens=self._snippet_max_tokens,
            highlight=self._highlight,
            highlight_prefix=self._highlight_prefix,
            highlight_suffix=self._highlight_suffix,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SQLiteBM25Retriever":
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

    def _create_bm25_index(self):
        """Creates the bm25 index table and triggers if they do not already exist."""

        cursor = self._document_store.db.cursor()
        cursor.execute("BEGIN TRANSACTION;")
        try:
            # TODO: Make this bit do something
            # If a config entry exists in the datasource, grab it
            # res = cursor.execute("SELECT config FROM retrievers WHERE id = ?", [self.NAME]).fetchone()
            # if res:
            #    config = json.loads(res[0])

            # Construct the full-text-search index
            tokenizer = self._tokenizer
            cursor.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS document_fts
                USING fts5("id", "content", tokenize = {tokenizer}, content='document', content_rowid='rowid');
            """
            )

            # Creates triggers that update the index when the documents are added/removed/updated
            cursor.execute(
                """
                CREATE TRIGGER IF NOT EXISTS document_ai_bm25 AFTER INSERT ON document BEGIN
                    INSERT INTO document_fts(rowid, content) VALUES (new.rowid, new.content);
                END;
                """
            )
            cursor.execute(
                """
                CREATE TRIGGER IF NOT EXISTS document_ad_bm25 AFTER DELETE ON document BEGIN
                  INSERT INTO document_fts(document_fts, rowid, content) VALUES('delete', old.rowid, old.content);
                END;
                """
            )
            cursor.execute(
                """
                CREATE TRIGGER IF NOT EXISTS document_au_bm25 AFTER UPDATE ON document BEGIN
                  INSERT INTO document_fts(document_fts, rowid, content) VALUES('delete', old.rowid, old.content);
                  INSERT INTO document_fts(rowid, content) VALUES (new.rowid, new.content);
                END;
                """
            )

            # Add the retrievers table
            config = self.to_dict()
            config = config["init_parameters"]
            if "document_store" in config:
                del config["document_store"]
            config_json = json.dumps(config)

            cursor.execute("""INSERT OR REPLACE INTO retrievers(id, config) VALUES(?, ?);""", [self.NAME, config_json])
        except sqlite3.Error as err:
            self._document_store.db.rollback()
            raise err
        finally:
            self._document_store.db.commit()
