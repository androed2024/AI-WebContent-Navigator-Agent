import os
import sys
import time
import requests
import torch
import tiktoken
import fitz  # PyMuPDF
import streamlit as st
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs
from collections import defaultdict
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright
from sentence_transformers import CrossEncoder
from supabase import create_client
from langdetect import detect
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI

# ────────────────────────────── Setup ──────────────────────────────
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")

MIN_CHUNK_TOKENS_HTML = int(os.getenv("MIN_CHUNK_TOKENS_HTML", "100"))
# BASE_URL = "https://www.lfk-online.de/pflegedienste.html/"

print("BASE URL", BASE_URL)

if not all([SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY]):
    st.error("Bitte SUPABASE_URL, SUPABASE_API_KEY und OPENAI_API_KEY setzen.")
    st.stop()

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_cli = OpenAI(api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2", activation_fn=torch.nn.Tanh()
)

# html_counter = pdf_counter = total_chunks = 0


# ────────────────────────────── Utils ──────────────────────────────
def is_pdf_like(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.path.lower().endswith(".pdf"):
        return True
    query = parse_qs(parsed.query)
    for val in query.values():
        for item in val:
            if ".pdf" in item.lower():
                return True
    return False


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
    print(f"🌐 Rufe Seite auf: {url}")
    try:
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
        clean_text = soup.get_text(separator="\n", strip=True)
        print("🔍 Extracted Text Preview:", repr(clean_text[:300]))
        return clean_text
    except Exception as e:
        print(f"❌ Fehler bei fetch_rendered_text_playwright: {e}")
        return ""


def extract_pdf_pages(data):
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        return [
            (i + 1, page.get_text().strip())
            for i, page in enumerate(doc)
            if page.get_text().strip()
        ]
    except Exception as e:
        print(f"❌ PDF-Parsing-Fehler: {e}")
        return []


def chunk_with_overlap(text, chunk_size=365, overlap=50, min_tokens=0):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if len(tokens) < min_tokens:
        print(
            f"⚠️ Zu wenig Tokens: {len(tokens)} (<{min_tokens}). Text (gekürzt): {repr(text[:120])}"
        )
        return []
    if len(tokens) < min_tokens:
        return []
    chunks = [
        enc.decode(tokens[i : i + chunk_size])
        for i in range(0, len(tokens) - chunk_size + 1, chunk_size - overlap)
    ]
    if not chunks and tokens:
        chunks = [enc.decode(tokens)]
    return chunks


def save_chunks(
    url, chunks, embeddings, page_number=None, source_page_url=None, anchor_text=None
):
    rows = []
    original_filename = os.path.basename(url) if is_pdf_like(url) else None

    for idx, (text, emb) in enumerate(zip(chunks, embeddings)):
        metadata = {
            "source_url": url,
            "source_page_url": source_page_url or url,
            "page": page_number or idx + 1,
            "chunk_index": idx + 1,
        }
        if anchor_text:
            metadata["anchor_text"] = anchor_text
        if original_filename:
            metadata["original_filename"] = original_filename

        rows.append({"content": text, "metadata": metadata, "embedding": emb})

    print(f"💾 Speichere {len(rows)} Chunks → {url}")
    supabase.table("lfk_rag_documents").insert(rows).execute()


def check_existing_data():
    try:
        response = supabase.table("lfk_rag_documents").select("metadata").execute()
        data = response.data or []
        html_urls = set()
        pdf_urls = set()
        for r in data:
            url = r.get("metadata", {}).get("source_url", "")
            if is_pdf_like(url):
                pdf_urls.add(url)
            else:
                html_urls.add(url)
        print(
            f"📦 Bereits in Supabase gespeichert: {len(html_urls)} HTML, {len(pdf_urls)} PDF"
        )
        return html_urls, pdf_urls
    except Exception as e:
        print(f"❌ Fehler bei check_existing_data(): {e}")
        return set(), set()


def find_links_recursive(
    base_url, max_depth=2, max_pdfs=10, visited=None, current_depth=0
):
    if visited is None:
        visited = set()
    html_links, pdf_links = set(), set()
    if current_depth > max_depth or base_url in visited:
        return html_links, pdf_links
    visited.add(base_url)
    try:
        print(f"🔍 Untersuche: {base_url}")
        resp = requests.get(base_url, timeout=10)
        soup = BeautifulSoup(resp.content, "html.parser")
        domain = urlparse(base_url).netloc

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            full_url = urljoin(base_url, href)

            if is_pdf_like(full_url) and len(pdf_links) < max_pdfs:
                print(f"🔗 PDF gefunden: {full_url} (von: {base_url})")
                anchor = a.get_text(strip=True) or None
                pdf_links.add((full_url, base_url, anchor))
                continue

            if urlparse(full_url).netloc != domain:
                continue
            if full_url.lower().endswith(".zip"):
                continue
            if full_url.startswith("http") and full_url not in visited:
                print(f"➡️ HTML-Link: {full_url}")
                html_links.add(full_url)
        for link in list(html_links):
            sub_html, sub_pdfs = find_links_recursive(
                link, max_depth, max_pdfs, visited, current_depth + 1
            )
            html_links.update(sub_html)
            pdf_links.update(sub_pdfs)
            if len(pdf_links) >= max_pdfs:
                break
    except Exception as e:
        print(f"⚠️ Fehler beim Laden von {base_url}: {e}")
    return html_links, pdf_links


def crawl():
    print("📢 Starte Crawling...")
    html_links, pdf_links = find_links_recursive(BASE_URL, max_depth=2)
    print(f"📊 Gefundene Seiten: {len(html_links)} HTML, {len(pdf_links)} PDF")
    html_saved = pdf_saved = chunk_count = 0

    # html Content
    for i, (html_url) in enumerate(sorted(html_links)):
        print(f"🌐 HTML-Seite {i + 1}: {html_url}")
        h, p, c = process_url(html_url, is_pdf=False)
        html_saved += h
        pdf_saved += p
        chunk_count += c

    # pdf content / files
    for i, (pdf_url, source_page_url, anchor_text) in enumerate(sorted(pdf_links)):
        print(f"📄 PDF-Datei {i + 1}: {pdf_url} (gefunden auf {source_page_url})")
        _, p, c = process_url(
            pdf_url,
            is_pdf=True,
            source_page_url=source_page_url,
            anchor_text=anchor_text,
        )
        pdf_saved += p
        chunk_count += c

    print("✅ Verarbeitung abgeschlossen:")
    print(f"   • HTML: {html_saved} Seiten")
    print(f"   • PDF:  {pdf_saved} Dateien")
    print(f"   • 🔹 {chunk_count} Chunks gespeichert")


def process_url(
    url: str, is_pdf: bool = False, source_page_url: str = None, anchor_text: str = None
) -> tuple[int, int, int]:
    print(f"🔍 Verarbeite URL: {url} (PDF: {is_pdf})")
    if is_pdf:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            pages = extract_pdf_pages(resp.content)
            all_chunks = [
                chunk for _, text in pages for chunk in chunk_with_overlap(text)
            ]
            if not all_chunks:
                print("⚠️ PDF hat keine Chunks.")
                return 0, 0, 0
            embeddings = openai_cli.embeddings.create(
                model="text-embedding-3-small", input=all_chunks
            )
            save_chunks(
                url,
                all_chunks,
                [d.embedding for d in embeddings.data],
                source_page_url=source_page_url,
                anchor_text=anchor_text,
            )

            print(f"✅ PDF gespeichert: {url}")
            return 0, 1, len(all_chunks)
        except Exception as e:
            print(f"❌ Fehler bei PDF: {e}")
            return 0, 0, 0
    else:
        try:
            text = fetch_rendered_text_playwright(url)
            if not text.strip():
                print("⚠️ HTML hat keinen sichtbaren Text.")
                return 0, 0, 0
            chunks = chunk_with_overlap(text, min_tokens=MIN_CHUNK_TOKENS_HTML)
            if not chunks:
                print("⚠️ HTML hat keine Chunks bzw <100 token.")
                return 0, 0, 0
            embeddings = openai_cli.embeddings.create(
                model="text-embedding-3-small", input=chunks
            )
            save_chunks(
                url, chunks, [d.embedding for d in embeddings.data], anchor_text=None
            )
            print(f"✅ HTML gespeichert: {url}")
            return 1, 0, len(chunks)
        except Exception as e:
            print(f"❌ Fehler bei HTML: {e}")
            return 0, 0, 0


# ▶️ Initialisierung
print("🚀 Starte Initialisierung...")
htmls, pdfs = check_existing_data()

if not htmls and not pdfs:
    print("🔁 Keine Einträge gefunden – starte Crawling...")
    crawl()
else:
    print("⏩ Daten bereits vorhanden – überspringe Crawling.")

# ▶️ Streamlit Chatbot UI
st.image("lfk_logo.jpg", width=150)
st.title("KI Web Assistent vom LfK")
with st.chat_message("assistant"):
    st.markdown(
        "**Hallo, ich bin der KI-Webassistent des LfK. Wie kann ich Ihnen helfen?**<br>"
        "Suchen Sie bestimmte Informationen auf unserer Webseite?",
        unsafe_allow_html=True,
    )

answer_mode = (
    "link-only"
    if st.toggle("Nur Link-Antwort anzeigen (Demo-Modus)", value=True)
    else "volltext"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chat_links" not in st.session_state:
    st.session_state.chat_links = []

question = st.chat_input("Frage eingeben:")
if question:
    with st.spinner("Suche läuft..."):
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = SupabaseVectorStore(
            client=supabase,
            embedding=embedding_model,
            table_name="lfk_rag_documents",
            query_name="match_rag_pages",
        )
        lang = detect(question)

        print(f"\n---[RAG Retrieval]---\nFrage: {question}")
        docs = vectorstore.similarity_search(question, k=15)
        print(f"🔎 similarity_search → {len(docs)} Treffer")
        for i, doc in enumerate(docs):
            snippet = doc.page_content[:120].replace("\n", " ")
            print(f"[{i+1}] {snippet} ... (URL: {doc.metadata.get('source_url')})")

        if not docs:
            answer = "Dazu habe ich leider keine Informationen gefunden, die auf unserer Webseite verfügbar sind."
            st.session_state.chat_history.append((question, answer))
            st.session_state.chat_links.append([])
        else:
            cross_input = [[question, doc.page_content] for doc in docs]
            scores = cross_encoder.predict(cross_input)
            for i, (doc, score) in enumerate(zip(docs, scores)):
                print(f"Score: {score:.3f} | {doc.metadata.get('source_url')}")
            threshold = 0.6
            scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            filtered_docs = [doc for doc, score in scored_docs if score >= threshold]

            if not filtered_docs and scored_docs:
                best_doc, best_score = scored_docs[0]
                print(
                    f"⚠️ Fallback aktiv: Score war {best_score:.3f}, aber bestes Doc wird genutzt."
                )
                filtered_docs = [best_doc]

            texts = [
                doc.page_content for doc in filtered_docs if doc.page_content.strip()
            ]

            if not texts:
                answer = "Dazu habe ich leider keine Informationen gefunden, die auf unserer Webseite verfügbar sind."
                st.session_state.chat_history.append((question, answer))
                st.session_state.chat_links.append([])
            else:
                context = "\n\n".join(texts)

                main_link = ""
                for doc in filtered_docs:
                    meta = doc.metadata
                    main_link = meta.get("source_page_url") or meta.get("source_url")
                    if main_link:
                        break

                if lang == "de":
                    if answer_mode == "link-only":
                        link_text = main_link if main_link else "unsere Webseite"

                        prompt = f"""
Ein Nutzer stellt dir eine Frage und möchte wissen, ob es dazu Inhalte auf unserer Webseite gibt.

Bitte beachte folgende Regeln:

1. Wenn es zu der Frage **relevante Inhalte auf der Webseite** gibt, antworte **immer genau so**:
"Ja, dazu gibt es Informationen auf unserer Webseite. Hier ist der passende Link: {link_text}."

2. Wenn es **keine passenden Inhalte** auf der Webseite gibt, dann antworte:
"Dazu habe ich leider keine Informationen gefunden, die auf unserer Webseite verfügbar sind."

Wichtige Hinweise:
- Du sollst **keinen** Inhalt zusammenfassen oder erklären.
- Gib **niemals** weitere Details oder Textauszüge aus dem Kontext wieder.
- Du antwortest **nur** mit einem der beiden Sätze oben, je nachdem ob der Kontext passt oder nicht.

Hier ist die Nutzerfrage:
{question}

Hier ist der verfügbare Kontext:
{context}
"""
                    else:
                        prompt = f"""
Du bist ein hilfreicher KI-Assistent. Beantworte die folgende Frage so gut wie möglich anhand des zur Verfügung gestellten Kontextes. Gib deine Antwort in gut verständlichem Deutsch und fasse den relevanten Inhalt zusammen. Antworte ausschließlich auf Basis des Kontextes. Gib keine erfundenen Informationen an.

Frage:
{question}

Kontext:
{context}
"""
                else:
                    prompt = f"Frage: {question}\n\nBitte antworte NUR auf der Grundlage des folgenden Inhalts:\n\n{context}"

                response = llm.invoke(prompt)
                st.session_state.chat_history.append(
                    (question, response.content.strip())
                )
                st.session_state.chat_links.append(
                    filtered_docs
                    if "keine Informationen" not in response.content
                    else []
                )

# Anzeige Chatverlauf inkl. Links
for idx, (question, answer) in enumerate(st.session_state.chat_history):
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        st.markdown(answer)

        docs = st.session_state.chat_links[idx]
        if docs:
            url_to_pages = defaultdict(list)
            url_to_pdfs = {}
            for doc in docs:
                meta = doc.metadata
                source_url = meta.get("source_url")
                display_url = meta.get("source_page_url") or source_url
                page = meta.get("page")
                if display_url:
                    url_to_pages[display_url].append(page)
                    url_to_pdfs[display_url] = source_url

            st.markdown("**Link:**")
            for url, pages in url_to_pages.items():
                pages_str = ", ".join(str(p) for p in sorted(set(pages)))
                st.markdown(
                    f'- <a href="{url}" target="_blank" rel="noopener noreferrer">{url}</a> '
                    f'(Seite{"n" if len(pages)>1 else ""} {pages_str})',
                    unsafe_allow_html=True,
                )
                pdf_url = url_to_pdfs.get(url)
                if pdf_url and is_pdf_like(pdf_url):
                    anchor_text = None
                    for doc in docs:
                        if doc.metadata.get("source_url") == pdf_url:
                            anchor_text = doc.metadata.get("anchor_text")
                            break
                    label = anchor_text or "PDF öffnen"
                    st.markdown(
                        f'<span title="PDF direkt öffnen" style="color: red">📄 <a href="{pdf_url}" target="_blank" rel="noopener noreferrer">{label}</a></span>',
                        unsafe_allow_html=True,
                    )
