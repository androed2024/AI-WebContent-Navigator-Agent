This Streamlit-based AI application recursively scrapes websites, extracts PDF and HTML content, and processes the text into vector embeddings stored in Supabase.
A built-in RAG (Retrieval-Augmented Generation) chatbot allows users to ask questions, with AI-generated answers referencing the original URLs.

---

supabase SQL command for RPC Function.
-- Make sure the pgvector extension is enabled
create extension if not exists vector;

-- Create the match_rag_pages RPC function
create or replace function match_rag_pages(
  query_embedding vector(1536),
  match_count int default 15
)
returns table (
  id uuid,
  content text,
  metadata jsonb,
  embedding vector(1536),
  similarity float
)
language sql stable
as $$
  select *,
    1 - (embedding <=> query_embedding) as similarity
  from lfk_rag_documents
  order by embedding <=> query_embedding
  limit match_count;
$$;


## Core Features

* Recursive crawling of a given base URL (depth-limited)
* PDF and HTML parsing with content extraction
* Chunking and embedding via text-embedding-3-small (OpenAI)
* Storage of chunks in Supabase (metadata, content, vector)
* Chatbot interface (LLM-based) to ask natural-language questions
* CrossEncoder reranking + score filtering for high-quality retrieval
* Output includes the original content sources (URLs)

---

## How It Works: Process Steps & Function Breakdown

### 1. `find_links_recursive(base_url, max_depth, max_pdfs, current_depth)`

This function scans the provided `BASE_URL` recursively to find all internal HTML pages and PDF files.

* **max\_depth**: limits how deep the crawler traverses from the base URL (default = 2)
* **max\_pdfs**: limits the number of PDF links gathered (default = 50)
* **current\_depth**: tracks recursion depth to prevent excessive crawling
* Skips external domains
* Returns two sets: HTML URLs and PDF URLs

### 2. `extract_pdf_pages(data)` + `chunk_with_overlap(text, chunk_size=365, overlap=50)`

These functions prepare the content for embedding:

* `extract_pdf_pages` parses the raw PDF content and extracts readable text from each page.
* `chunk_with_overlap` tokenizes the text (HTML or PDF) using the `tiktoken` encoder ("cl100k\_base"), and slices it into overlapping chunks:

  * **chunk\_size**: 365 tokens
  * **overlap**: 50 tokens
  * This overlap preserves context across chunk boundaries.
* Embeddings are created using OpenAI's `text-embedding-3-small` and stored in Supabase (`lfk_rag_documents` table).

### 3. `ask_question(question: str)`

This function powers the chatbot interface for user queries:

* Uses `langdetect` to auto-detect question language
* Performs **semantic similarity search** via SupabaseVectorStore with `k=15` top chunks
* Applies **CrossEncoder reranking** with:

  * Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
  * Activation: `Sigmoid` to normalize relevance score \[0.0–1.0]
* **Threshold filtering**: discards results with score < `0.6`
* Aggregates filtered chunks as context and feeds to `ChatOpenAI` to generate answer
* Final output includes answer and up to 5 top-matching source URLs with score display

---

## Run the App

```bash
streamlit run main.py
```

---

## Create Supabase Database

Make sure your Supabase project includes the following:

```sql
create table lfk_rag_documents (
  id uuid primary key default gen_random_uuid(),
  content text not null,
  metadata jsonb,
  embedding vector(1536), -- OpenAI "text-embedding-3-small" has 1536 dimensions
  created_at timestamp with time zone default now()
);

-- Also required:
create extension if not exists vector;
```

---

## Requirements

Make sure to install required packages via:

```bash
pip install -r requirements.txt
playwright install  # Needed once to install Chromium headless browser
```

---

## Notes

* `.env` file must define:

  * `SUPABASE_URL`
  * `SUPABASE_API_KEY`
  * `OPENAI_API_KEY`
  * `BASE_URL`
* PDF and HTML content must contain text to be embedded (e.g. image-based PDFs will be skipped)
* Crawler avoids external domains automatically
* To speed up processing during development, set:

  * `max_depth=1`
  * `max_pdfs=5`

---

Feel free to fork, improve, or extend this RAG-based knowledge agent for your own structured content use cases.
# AI-WebContent-Navigator-Agent
