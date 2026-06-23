# Manchester United Wikipedia RAG

A small portfolio RAG project that treats the subsections of the Manchester United Wikipedia page as documents.

## What it does

1. Fetches the Manchester United F.C. page from Wikipedia using the MediaWiki API.
2. Splits the page into section-level documents.
3. Cleans the HTML into plain text.
4. Embeds each section using `sentence-transformers/all-MiniLM-L6-v2`.
5. Retrieves the most relevant sections for a user question.
6. Produces a grounded extractive answer and displays source sections with similarity scores.

## Why this is useful for a portfolio

This shows the main parts of a RAG pipeline:

- data ingestion
- document cleaning
- document indexing
- embedding generation
- semantic retrieval
- answer generation
- source attribution
- basic user interface

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source venv/bin/activate
```

## Run the command line app

```bash
python app.py
```

## Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

## Example questions

```text
When did Manchester United move to Old Trafford?
Why are Manchester United called the Red Devils?
What happened in the Munich air disaster?
Who was Sir Alex Ferguson and why was he important?
What are Manchester United's main rivalries?
What trophies has Manchester United won?
```

## Next improvements

- Add an LLM call for abstractive answer generation.
- Store embeddings in Chroma or FAISS.
- Add automatic evaluation questions and expected source sections.
- Add a Dockerfile.
- Add a small deployment on Streamlit Community Cloud.
- Expand from one page to multiple club pages.

## Attribution

The text is retrieved dynamically from Wikipedia. If you store or publish extracted article text, include Wikipedia attribution and comply with the page licence.
