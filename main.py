import os
from dotenv import load_dotenv
import time
import requests

import torch
from sentence_transformers import CrossEncoder

from urllib.parse import urljoin
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import tiktoken
import streamlit as st
from openai import OpenAI
from supabase import create_client
from langdetect import detect
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import requests
from sentence_transformers import CrossEncoder

# Load environment variables
load_dotenv()

# Required environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
BASE_URL = "BASE_URL = "https://www.lfk-online.de/pflegedienste/downloads/existenzgruender.html"

print("BASE-URL", BASE_URL)

# Validate environment configuration
if not all([SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY]):
    st.error(
        "Lütfen SUPABASE_URL, SUPABASE_API_KEY ve OPENAI_API_KEY ortam değişkenlerini ayarlayın."
    )
    st.stop()

# Initialize clients
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_cli = OpenAI(api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9)

cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    default_activation_function=torch.nn.Sigmoid(),
)


html_counter = 0
pdf_counter = 0


def fetch_iframe_content(page):
    iframe = page.query_selector("iframe")
    if not iframe:
        return None
    frame = iframe.content_frame()
    if not frame:
        return None
    frame.wait_for_load_state("networkidle")
    html = frame.content()
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def fetch_rendered_text_playwright(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=60000)
        page.wait_for_load_state("networkidle")
        text = fetch_iframe_content(page) or page.content()
        browser.close()
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def extract_pdf_pages(data):
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                pages.append((i + 1, text))
        return pages
    except Exception:
        return []


def chunk_with_overlap(text, chunk_size=365, overlap=50):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = [
        enc.decode(tokens[i : i + chunk_size])
        for i in range(0, len(tokens) - chunk_size + 1, chunk_size - overlap)
    ]
    return chunks


# parse websites. Maximum depth and max pdf (to limit token cost for test)
def find_links_recursive(
    base_url, max_depth=0, max_pdfs=10, visited=None, current_depth=0
):
    if visited is None:
        visited = set()

    html_links = set()
    pdf_links = set()

    if current_depth > max_depth or base_url in visited:
        return html_links, pdf_links

    visited.add(base_url)

    try:
        resp = requests.get(base_url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        domain = urlparse(base_url).netloc

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            full_url = urljoin(base_url, href)
            if urlparse(full_url).netloc != domain:
                continue  # skip external links

            if full_url.lower().endswith(".pdf"):
                if len(pdf_links) < max_pdfs:
                    pdf_links.add(full_url)
            elif full_url.startswith("http") and full_url not in visited:
                html_links.add(full_url)

        # Rekursiv weitere HTML-Links durchlaufen
        for link in list(html_links):
            sub_html, sub_pdfs = find_links_recursive(
                link,
                max_depth=max_depth,
                max_pdfs=max_pdfs,
                visited=visited,
                current_depth=current_depth + 1,
            )
            html_links.update(sub_html)
            pdf_links.update(sub_pdfs)
            if len(pdf_links) >= max_pdfs:
                break

    except Exception as e:
        print(f"⚠️ 404 or fetch failed: {base_url} → {str(e)}")

    return html_links, pdf_links


def save_chunks(url, chunks, embeddings, page_number=None):
    rows = []
    for idx, (text, emb) in enumerate(zip(chunks, embeddings)):
        metadata = {
            "source_url": url,
            "page": page_number or idx + 1,
            "chunk_index": idx + 1,
        }
        rows.append({"content": text, "metadata": metadata, "embedding": emb})
    supabase.table("lfk_rag_documents").insert(rows).execute()


def process_url(url, is_pdf=False):
    global html_counter, pdf_counter
    try:
        if is_pdf:
            pdf_counter += 1
            print(f"📄 PDF ({pdf_counter}): {url}")
            resp = requests.get(url, timeout=10)
            if resp.status_code == 404:
                print(f"⚠️ Link nicht gefunden (404): {base_url}")
                return html_links, pdf_links
            resp.raise_for_status()
            pages = extract_pdf_pages(resp.content)

            if not pages:
                print(f"⚠️ PDF enthält keinen lesbaren Text: {url}")
                return

            for pg, text in pages:
                chunks = chunk_with_overlap(text)
                if not chunks:
                    continue
                embeddings = openai_cli.embeddings.create(
                    model="text-embedding-3-small", input=chunks
                )
                save_chunks(
                    url, chunks, [d.embedding for d in embeddings.data], page_number=pg
                )

        else:
            html_counter += 1
            print(f"🌐 HTML ({html_counter}): {url}")
            text = fetch_rendered_text_playwright(url)
            if not text.strip():
                print(f"⚠️ Leere HTML-Seite oder kein sichtbarer Text: {url}")
                return

            chunks = chunk_with_overlap(text)
            if not chunks:
                print(f"⚠️ Keine Chunks erzeugt: {url}")
                return

            embeddings = openai_cli.embeddings.create(
                model="text-embedding-3-small", input=chunks
            )
            save_chunks(url, chunks, [d.embedding for d in embeddings.data])

        print(f"✅ Verarbeitung abgeschlossen: {url}")

    except Exception as e:
        st.warning(f"❌ Fehler bei {url}: {e}")
        print(f"[ERROR] {url}: {e}")


@st.cache_resource(show_spinner=False)
def init_vectorstore_and_data():
    htmls, pdfs = find_links_recursive(BASE_URL, max_depth=2, max_pdfs=50)
    for url in htmls:
        process_url(url, is_pdf=False)
        time.sleep(1)
    for url in pdfs:
        process_url(url, is_pdf=True)
        time.sleep(1)
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = SupabaseVectorStore(
        client=supabase,
        embedding=embedding_model,
        table_name="lfk_rag_documents",
        query_name="match_rag_pages",
    )
    print(f"✅ Gesamt verarbeitet: {html_counter} HTML-Seiten, {pdf_counter} PDFs")
    return vs


# Initialize vector store
try:
    vectorstore = init_vectorstore_and_data()
except Exception as e:
    st.error(f"Error initializing data: {e}")
    st.stop()


def ask_question(question: str):
    lang = detect(question)
    if lang not in ["en", "de"]:
        return ("Please ask a clear and detailed question in English or German.", [])

    docs = vectorstore.similarity_search(question, k=15)

    # CrossEncoder-Reranking
    cross_input = [[question, doc.page_content] for doc in docs]
    scores = cross_encoder.predict(cross_input)

    # Score-basierte Filterung
    threshold = 0.6
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    filtered_docs = [doc for doc, score in scored_docs if score >= threshold]

    for doc, score in scored_docs[:5]:
        print(f"Score: {score:.3f} | URL: {doc.metadata.get('source_url')}")

    texts = [doc.page_content for doc in filtered_docs if doc.page_content.strip()]

    if not texts:
        return ("No sufficient information available.", [])
    context = "\n\n".join(texts)
    if lang == "de":
        prompt = f"""
Frage: {question}

Bitte nur basierend auf dem folgenden Text antworten:

{context}

"""
    else:
        prompt = f"""
Question: {question}

Please answer ONLY based on the following content:

{context}

"""
    response = llm.invoke(prompt)
    answer = response.content.strip()
    return answer, scored_docs


# Streamlit UI
st.title("RAG Chatbot")
question = st.text_input("Enter your question:")
if question:
    with st.spinner("Searching..."):
        ans, scored_docs = ask_question(question)
    st.subheader("Answer:")
    st.write(ans)
    if sources:
        st.subheader("Sources:")
        for doc, score in scored_docs[:5]:
            meta = doc.metadata
            st.write(
                f"- {meta.get('source_url')} (Page {meta.get('page')}, Score: {score:.2f})"
            )
