# SQLite Haystack Document store
[![test](https://github.com/alanmeeson/sqlite-haystack/actions/workflows/test.yml/badge.svg)](https://github.com/alanmeeson/sqlite-haystack/actions/workflows/test.yml)

SQLite-Haystack is an embedded [SQLite](https://sqlite.org) backed Document Store for 
[Haystack 2.X](https://github.com/deepset-ai/haystack/) intended for prototyping or small systems.  

Currently supported features are:
- Embedded database for in memory or on disk use.
- BM25 Free Text Search using [SQLite FTS5](https://sqlite.org/fts5.html)
- Bring-your-own-embedding vector search using [SQLite](https://github.com/asg017/sqlite-vss)

## Installation

The current simplest way to get SQLite-Haystack is to install from GitHub via pip:

```pip install git+https://github.com/alanmeeson/sqlite-haystack.git```

## Usage

See the example in `examples/pipeline-usage.ipynb`

## Development

### Test

You can use `hatch` to run the linters:

```console
~$ hatch run lint:all
cmd [1] | ruff .
cmd [2] | black --check --diff .
All done! ✨ 🍰 ✨
6 files would be left unchanged.
cmd [3] | mypy --install-types --non-interactive src/sqlite_haystack tests
Success: no issues found in 6 source files
```

Similar for running the tests:

```console
~$ hatch run cov
cmd [1] | coverage run -m pytest tests
...
```

### Build

To build the package you can use `hatch`:

```console
~$ hatch build
[sdist]
dist/sqlite_haystack-0.0.1.tar.gz

[wheel]
dist/sqlite_haystack-0.0.1-py3-none-any.whl
```

### Roadmap

In no particular order:
- **Address the 1GB limit to the vector index size.**
  
  This is a limitation of the sqlite-vss module being used for vector search, so see the
  [issue in that repo](https://github.com/asg017/sqlite-vss/issues/1) for details. It only applies to the embeddings, 
  so as a rough estimate, if using openAI's current largest embedding model text-embedding-3-large you could index 
  approximately 85K Documents.  If using something smaller, like all-MiniLM-L6-v2 you could index approximately 600K 
  Documents.

- **Write more documentation**
  
  There's docstrings in the code, but there's an outstanding task to configure pydoc or similar to produce some api
  docs.  Also, documentation to explain the data model and how it works would be a good idea.

- **Add more tests on the embedding and bm25 retriever functions.**
  
  Most of the testing is the standard DocumentStore tests from the Haystack template, with some added tests of the 
  retrievers adapted from the ElasticSearch components.  These don't cover the retriever code as much as I would like,
  though the example notebook does serve as a simple system test script.

- **Add optional index creation on the metadata fields for faster filtering.**
  
  The metadata is stored in a JSON field in the main documents table.  To speed up queries, we could add a method to 
  allow users to create an index for the metadata elements which are commonly filtered on, as covered in 
  [this article](https://sqldocs.org/sqlite/sqlite-json-data/).  For bonus points, we could keep a track of the fields
  used for filtering, and automatically create indexes for fields that are used over a certain number of times.

- **Make the tokenizer used in the FTS/BM25 index configurable**
  
  Currently the FTS is using a plain `trigram` tokeniser.  There are other options, such as porter stemming, and further
  tricks that can be applied as part of the FTS process.  Ideally, we want to make this user configurable, so that users
  can tailor the method used to their use case.

- **Investigate making the VSS configurable.**

  Similarly, for the vss_search step,  we currently only provide a limit on the number of documents to retrieve.  There
  are probably more configuration options worth exposing.

## Limitations

- 1GB limit to the vector index size.
- Not available for Windows:  The sqlite-vss package currently doesn't have a windows release, so neither does this.