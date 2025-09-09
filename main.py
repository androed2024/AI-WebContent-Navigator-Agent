# -*- coding: utf-8 -*-
"""
LFK KI-Webassistent 
===================

Archiketur:


Wichtige Hinweise
-----------------
- Der Code crawlt aktuell **HTML** und **PDF**. Die Office-Formate werden bewusst **noch nicht**
  geparst – TODO-Sektionen sind mit "OFFICE-PARSER" markiert.
- Scraping-Tiefe, Limits, Seeds etc. kommen aus **.env**.
- Embeddings: **text-embedding-3-small**. Speicherung in **lfk_rag_documents**.
- Retrieval via **Supabase RPC** (match_lfk_rag_documents) + **Similarity Search**
  (CrossEncoder Re-Ranking entfernt für erste Produktionsversion).
- Streamlit-UI mit *Link-only*-Demo und *Volltext*-Antwortmodus.
"""

import os
import sys
import time
import requests
import tiktoken
import fitz  # PyMuPDF
import streamlit as st
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs
import re
from collections import defaultdict
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright
# Reranking entfernt – kann später optional wieder aktiviert werden
# from sentence_transformers import CrossEncoder
# import torch
from supabase import create_client
from langdetect import detect
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI
from urllib.parse import urlparse

# Import scraping functions from separate module
from scraping import (
    norm_url,
    is_pdf_like,
    looks_like_media,
    head_is_pdf,
    should_probe_via_head,
    fetch_rendered_text_playwright,
    extract_structured_html_sections,
    extract_pdf_pages,
    fetch_pdf,
    find_links_recursive_rendered,
    process_url,
    init_portal_auth,
    http_get,
    http_head,
    domains_match,
    normalize_domain
)

# ────────────────────────────── Setup ──────────────────────────────
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")

MIN_CHUNK_TOKENS_HTML = int(os.getenv("MIN_CHUNK_TOKENS_HTML", "100"))

# Seeds, limits, Priorisierung
MAX_DEPTH = int(os.getenv("MAX_DEPTH", "1"))
MAX_PDFS = int(os.getenv("MAX_PDFS", "5"))
MAX_HTML = int(os.getenv("MAX_HTML", "10"))

PRIORITY_HTML_KEYWORDS = (
    "downloads",
    "gratis_downloads",
    "dokumente",
    "downloadbereich",
    "mediathek",
    "existenzgruender",
    "fileadmin",
    "pdf",
)

BASE_URL = (os.getenv("BASE_URL") or "").rstrip("/")
EXTRA_SEEDS = (os.getenv("EXTRA_SEEDS") or "").split()

SEED_URLS = [BASE_URL] + EXTRA_SEEDS if BASE_URL else EXTRA_SEEDS

MEDIA_EXTS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".svg",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".mp3",
    ".wav",
)

# Einige Server liefern PDFs nur mit speziellen Headern / nach Redirects
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "*/*",
}

# Reranking entfernt – kann später optional wieder aktiviert werden
# CROSSENCODER_THRESHOLD = float(os.getenv("CROSSENCODER_THRESHOLD", "0.55"))
# SCORE_GAP = float(os.getenv("SCORE_GAP", "0.08"))
# MIN_SECOND_SCORE = float(os.getenv("MIN_SECOND_SCORE", "0.6"))

# Similarity Search Konfiguration
SIMILARITY_SEARCH_K = int(os.getenv("SIMILARITY_SEARCH_K", "30"))
MAX_DISPLAY_LINKS = int(os.getenv("MAX_DISPLAY_LINKS", "2"))

print("BASE URL", BASE_URL)

if not all([SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY]):
    st.error("Bitte SUPABASE_URL, SUPABASE_API_KEY und OPENAI_API_KEY setzen.")
    st.stop()

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_cli = OpenAI(api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.9)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
# Reranking entfernt – kann später optional wieder aktiviert werden
# cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", activation_fn=torch.nn.Tanh())


# ────────────────────────────── Utils / URL & Heuristiken ──────────────────────────────


def _priority_tuple_for_html(url: str, existing_htmls: set[str]) -> tuple:
    """Ranking-Schlüssel für HTML-Links.
    - bekannte URLs später abarbeiten (True > False)
    - Links mit Prioritäts-Keywords früher abarbeiten
    - lexikographische Tiebreaker
    """
    u = (url or "").lower()
    is_existing = (norm_url(url) in existing_htmls)
    has_keyword = any(k in u for k in PRIORITY_HTML_KEYWORDS)
    return (is_existing, not has_keyword, u)


# ────────────────────────────── Rendern & HTML‑Parsing ──────────────────────────────
# (Functions moved to scraping.py)


# ────────────────────────────── Text & Embeddings ──────────────────────────────
# (PDF extraction moved to scraping.py)


def chunk_with_overlap(text, chunk_size=365, overlap=50, min_tokens=0):
    """Token-basiertes Sliding-Window-Chunking (cl100k_base Encoding)."""
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if len(tokens) < min_tokens:
        print(
            f"⚠️ Zu wenig Tokens: {len(tokens)} (<{min_tokens}). Text (gekürzt): {repr(text[:120])}"
        )
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
    """Schreibt Chunks + Embeddings nach Supabase (Tabelle: lfk_rag_documents).
    Metadaten enthalten u. a. Normalized `source_url`, `source_url_raw`, `source_page_url`,
    Seiten-/Chunk-Indizes sowie optional `anchor_text` und `original_filename`.
    """
    rows = []
    original_filename = os.path.basename(url) if is_pdf_like(url) else None

    for idx, (text, emb) in enumerate(zip(chunks, embeddings)):
        metadata = {
            "source_url": norm_url(url),
            "source_url_raw": url,
            "source_page_url": norm_url(source_page_url or url),
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
    """Liest vorhandene `source_url`s aus Supabase, getrennt in HTML- und PDF-Mengen.
    Achtung: PDFs werden über Heuristik (URL oder `original_filename`) erkannt.
    """
    try:
        response = supabase.table("lfk_rag_documents").select("metadata").execute()
        data = response.data or []
        html_urls = set()
        pdf_urls = set()
        for r in data:
            md = r.get("metadata", {}) or {}
            raw = md.get("source_url_raw") or md.get("source_url") or ""
            url_n = norm_url(raw)
            is_pdf = is_pdf_like(raw) or str(
                md.get("original_filename", "")
            ).lower().endswith(".pdf")
            if is_pdf:
                pdf_urls.add(url_n)
            else:
                html_urls.add(url_n)
        print(
            f"📦 Bereits in Supabase gespeichert: {len(html_urls)} HTML, {len(pdf_urls)} PDF"
        )
        if html_urls:
            print("   📄 Bereits gespeicherte HTML-Seiten:")
            for i, url in enumerate(sorted(html_urls), 1):  # Zeige alle
                print(f"      [{i}] {url}")
        if pdf_urls:
            print("   📋 Bereits gespeicherte PDF-Dokumente:")
            for i, url in enumerate(sorted(pdf_urls), 1):  # Zeige alle
                print(f"      [{i}] {url}")
        return html_urls, pdf_urls
    except Exception as e:
        print(f"❌ Fehler bei check_existing_data(): {e}")
        return set(), set()


# (Regex patterns moved to scraping.py)


# ────────────────────────────── Crawler (gerendert) ──────────────────────────────
# (Functions moved to scraping.py)


# ────────────────────────────── Crawl-Steuerung ──────────────────────────────


def crawl():
    """Top-Level-Crawl über alle Seeds.
    - ruft `find_links_recursive_rendered` pro Seed
    - verarbeitet neue HTML- und PDF-Ziele via `process_url`
    - respektiert MAX_HTML/MAX_PDFS
    - schreibt Embeddings nach Supabase
    """
    print("📢 Starte Crawling...")
    print(f"🌱 Seeds: {SEED_URLS}")
    print(f"🌱 MAX_DEPTH: {MAX_DEPTH}, MAX_HTML: {MAX_HTML}, MAX_PDFS: {MAX_PDFS}")
    html_links, pdf_links = set(), set()
    visited = set()  # Neues visited Set nur für Crawling
    probe_cache = {}
    print("🧹 Neues visited Set für Crawling erstellt (Login-URLs werden nicht berücksichtigt)")

    for seed in SEED_URLS:
        s = norm_url(seed)
        if s in visited:
            continue
        print(f"🌱 Verarbeite Seed: {s}")
        h, p = find_links_recursive_rendered(
            s,
            max_depth=MAX_DEPTH,
            max_pdfs=MAX_PDFS,
            max_html=MAX_HTML,
            priority_html_keywords=PRIORITY_HTML_KEYWORDS,
            headers=HEADERS,
            visited=visited,
            current_depth=0,
            _probe_cache=probe_cache,
        )
        print(f"🌱 Seed {s} ergab: {len(h)} HTML, {len(p)} PDF")
        visited.add(s)  # Markiere als besucht nach erfolgreichem Crawling
        html_links.update(h)
        pdf_links.update(p)
        if len(html_links) >= MAX_HTML or len(pdf_links) >= MAX_PDFS:
            break

    print(f"📊 Gefundene Seiten: {len(html_links)} HTML, {len(pdf_links)} PDF")
    if html_links:
        print("   📄 HTML-Seiten:")
        for i, url in enumerate(sorted(html_links), 1):  # Zeige alle
            print(f"      [{i}] {url}")
    if pdf_links:
        print("   📋 PDF-Dokumente:")
        for i, (url, title, _) in enumerate(sorted(pdf_links), 1):  # Zeige alle
            print(f"      [{i}] {url} ({title or 'Unbekannt'})")

    html_saved = pdf_saved = chunk_count = 0
    existing_htmls, existing_pdfs = check_existing_data()

    # HTML-Queue: Medien raus, priorisiert sortieren
    all_html = [u for u in html_links if not looks_like_media(u)]
    all_html.sort(key=lambda u: _priority_tuple_for_html(u, existing_htmls))

    saved_html = 0
    skipped_html = 0
    processed_html_urls = []
    for html_url in all_html:
        if saved_html >= MAX_HTML:
            break
        if norm_url(html_url) in existing_htmls:
            skipped_html += 1
            continue
        print(f"🌐 HTML-Seite {saved_html + 1}: {html_url}")
        h, p, c = process_url(
            html_url, 
            is_pdf=False,
            openai_cli=openai_cli,
            save_chunks_func=save_chunks,
            chunk_with_overlap_func=chunk_with_overlap,
            min_chunk_tokens_html=MIN_CHUNK_TOKENS_HTML,
            headers=HEADERS
        )
        if h:
            saved_html += h
        processed_html_urls.append(html_url)
        pdf_saved += p
        chunk_count += c

    # PDFs priorisieren (fileadmin bevorzugt)
    saved_pdf = 0
    skipped_pdf = 0
    processed_pdf_urls = []
    pdf_list = sorted(
        pdf_links, key=lambda t: ("fileadmin" not in (t[0] or "").lower(), t[0])
    )
    for pdf_url, source_page_url, anchor_text in pdf_list:
        if saved_pdf >= MAX_PDFS:
            break
        if norm_url(pdf_url) in existing_pdfs:
            skipped_pdf += 1
            continue
        print(
            f"📄 PDF-Datei {saved_pdf + 1}: {pdf_url} (gefunden auf {source_page_url})"
        )
        _, p, c = process_url(
            pdf_url,
            is_pdf=True,
            source_page_url=source_page_url,
            anchor_text=anchor_text,
            openai_cli=openai_cli,
            save_chunks_func=save_chunks,
            chunk_with_overlap_func=chunk_with_overlap,
            headers=HEADERS
        )
        if p:
            saved_pdf += p
        processed_pdf_urls.append((pdf_url, anchor_text or "Unbekannt"))
        chunk_count += c

    print("✅ Verarbeitung abgeschlossen:")
    print(f"   • HTML: {saved_html} neu verarbeitet, {skipped_html} bereits vorhanden")
    if processed_html_urls:
        print("     📄 Neu verarbeitete HTML-Seiten:")
        for i, url in enumerate(processed_html_urls, 1):
            print(f"        [{i}] {url}")
    print(f"   • PDF: {saved_pdf} neu verarbeitet, {skipped_pdf} bereits vorhanden")
    if processed_pdf_urls:
        print("     📋 Neu verarbeitete PDF-Dokumente:")
        for i, (url, title) in enumerate(processed_pdf_urls, 1):
            print(f"        [{i}] {url} ({title})")
    print(f"   • 🔹 {chunk_count} neue Chunks gespeichert")


# ────────────────────────────── Verarbeitung pro URL ──────────────────────────────
# (process_url function moved to scraping.py)



# ────────────────────────────── Initialisierung ──────────────────────────────
if BASE_URL:
    BASE_URL = norm_url(BASE_URL)
print("🚀 Starte Initialisierung...")

# Authentifizierung initialisieren (falls Login-Daten vorhanden)
SCRAPE_USER = os.getenv("SCRAPE_USER")
SCRAPE_PASS = os.getenv("SCRAPE_PASS") 
LOGIN_URL = os.getenv("SCRAPE_LOGIN_URL") or os.getenv("LOGIN_URL")

if SCRAPE_USER and SCRAPE_PASS:
    print("🔐 Initialisiere Authentifizierung...")
    print(f"🔐 Login mit Benutzer: {SCRAPE_USER}")
    authenticated_session = init_portal_auth(
        username=SCRAPE_USER,
        password=SCRAPE_PASS, 
        login_url=LOGIN_URL,
        test_url=BASE_URL,
        headers=HEADERS
    )
else:
    print("⚠️ Keine Login-Daten gefunden - verwende unauthentifizierte Session")

htmls, pdfs = check_existing_data()

# Seeds generisch: nur aus .env (BASE_URL + OPTIONAL EXTRA_SEEDS)
SEED_URLS = [BASE_URL] + EXTRA_SEEDS if BASE_URL else EXTRA_SEEDS

if not htmls or not pdfs:
    print("🔁 Fehlende Datentypen → starte Crawling…")
    crawl()
else:
    print("⏩ Daten vorhanden – überspringe Crawling.")


# ────────────────────────────── Streamlit Chatbot UI ──────────────────────────────
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
        # Vectorstore + Sprachdetektion
        vectorstore = SupabaseVectorStore(
            client=supabase,
            embedding=embedding_model,
            table_name="lfk_rag_documents",
            query_name="match_lfk_rag_documents",
        )
        lang = detect(question)

        print(f"\n---[RAG Retrieval]---\nFrage: {question}")

        # 1) Direkter RPC-Debug (Ground Truth Sicht auf die Datenbank)
        q_emb = embedding_model.embed_query(question)
        rpc = supabase.rpc(
            "match_lfk_rag_documents",
            {"query_embedding": q_emb, "match_count": 5, "min_similarity": 0.2},
        ).execute()
        print("🔧 RPC rows:", len(rpc.data) if rpc.data else 0)
        for r in rpc.data or []:
            print(f"{r['similarity']:.3f}", (r["metadata"] or {}).get("source_url"))

        # 2) LangChain Similarity Search (k=SIMILARITY_SEARCH_K) – basiert nur auf Embedding-Ähnlichkeit
        docs = vectorstore.similarity_search(question, k=SIMILARITY_SEARCH_K)
        print(f"🔎 similarity_search → {len(docs)} Treffer")
        for i, doc in enumerate(docs):
            snippet = doc.page_content[:120].replace("\n", " ")
            print(f"[{i+1}] {snippet} ... (URL: {doc.metadata.get('source_url')})")

        if not docs:
            st.session_state.chat_history.append(
                (
                    question,
                    "Dazu habe ich leider keine Informationen gefunden, die auf unserer Webseite verfügbar sind.",
                ),
            )
            st.session_state.chat_links.append([])
        else:
            # Reranking entfernt – verwende direkt Similarity Search Reihenfolge
            print("📊 Verwende Similarity Search Ranking (ohne CrossEncoder)")
            
            # Kontext für Antwort: Top-3 Dokumente
            filtered_docs = docs[:3]
            
            # PDF priorisieren für Kontext
            best_pdf_doc = None
            for doc in docs:
                src = (doc.metadata or {}).get("source_url", "")
                if src and is_pdf_like(src):
                    best_pdf_doc = doc
                    break
            if best_pdf_doc and best_pdf_doc not in filtered_docs:
                filtered_docs = [best_pdf_doc] + filtered_docs[:2]

            # Anzeige-Links (max MAX_DISPLAY_LINKS) – PDFs bevorzugen
            candidates = docs[:5]  # Top-5 als Kandidaten
            first = None
            # Suche nach erstem PDF
            for doc in candidates:
                if is_pdf_like((doc.metadata or {}).get("source_url", "")):
                    first = doc
                    break
            # Falls kein PDF, nimm Top-1
            if not first and candidates:
                first = candidates[0]
            display_docs = [first] if first else []

            # Zweiten Link hinzufügen falls gewünscht
            if len(display_docs) < MAX_DISPLAY_LINKS and len(candidates) > 1:
                for doc in candidates:
                    if first and doc is first:
                        continue
                    display_docs.append(doc)
                    break
            display_docs = display_docs[:MAX_DISPLAY_LINKS]

            # Hauptlink bestimmen – bevorzugt PDF
            best_doc = docs[0]  # Top-1 aus Similarity Search
            best_meta = best_doc.metadata or {}
            main_link = best_meta.get("source_url") or best_meta.get("source_page_url")
            if not is_pdf_like(main_link):
                # Suche nach PDF in Top-Dokumenten
                for doc in docs[:5]:
                    m = doc.metadata or {}
                    cand = m.get("source_url")
                    if cand and is_pdf_like(cand):
                        main_link = cand
                        break
            if not main_link:
                for doc in filtered_docs:
                    meta = doc.metadata or {}
                    main_link = meta.get("source_page_url") or meta.get("source_url")
                    if main_link:
                        break

            # Antwort-Prompting (Link-only oder Volltext)
            if lang == "de":
                if answer_mode == "link-only":
                    link_text = main_link if main_link else "unsere Webseite"
                    context = "\n\n".join(doc.page_content for doc in filtered_docs)
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
                    context = "\n\n".join(doc.page_content for doc in filtered_docs)
                    prompt = f"""
            Du bist ein hilfreicher KI-Assistent. Beantworte die folgende Frage so gut wie möglich anhand des zur Verfügung gestellten Kontextes. Gib deine Antwort in gut verständlichem Deutsch und fasse den relevanten Inhalt zusammen. Antworte ausschließlich auf Basis des Kontextes. Gib keine erfundenen Informationen an.

            Frage:
            {question}

            Kontext:
            {context}
            """
            else:
                context = "\n\n".join(doc.page_content for doc in filtered_docs)
                prompt = f"Frage: {question}\n\nBitte antworte NUR auf der Grundlage des folgenden Inhalts:\n\n{context}"

            response = llm.invoke(prompt)
            st.session_state.chat_history.append((question, response.content.strip()))
            st.session_state.chat_links.append(
                display_docs if "keine Informationen" not in response.content else []
            )

# Anzeige Chatverlauf inkl. Links
for idx, (question, answer) in enumerate(st.session_state.chat_history):
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        st.markdown(answer)

        docs = st.session_state.chat_links[idx]
        if docs:

            def norm_pages(pages_set):
                return (
                    ", ".join(str(p) for p in sorted(pages_set)) if pages_set else "–"
                )

            groups = defaultdict(
                lambda: {
                    "pages": set(),
                    "is_pdf": False,
                    "anchor": None,
                    "filename": None,
                }
            )

            for doc in docs:
                m = doc.metadata or {}
                src = m.get("source_url")
                page_url = m.get("source_page_url") or src
                page = m.get("page")

                if src and is_pdf_like(src):
                    key = src
                    g = groups[key]
                    g["is_pdf"] = True
                    g["filename"] = m.get("original_filename")
                    if m.get("anchor_text"):
                        g["anchor"] = m.get("anchor_text")
                    if page is not None:
                        g["pages"].add(page)
                else:
                    key = page_url
                    g = groups[key]
                    if page is not None:
                        g["pages"].add(page)
            st.markdown("**Quellen:**")
            items = list(groups.items())
            items.sort(key=lambda kv: (not kv[1]["is_pdf"]))  # PDFs zuerst
            for url, info in items:
                pages_str = norm_pages(info["pages"])
                label = info["anchor"] or info["filename"] or url
                if info["is_pdf"]:
                    st.markdown(
                        f'- 📄 <a href="{url}" target="_blank" rel="noopener noreferrer">{label}</a> (Seiten {pages_str})',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'- <a href="{url}" target="_blank" rel="noopener noreferrer">{url}</a> (Seiten {pages_str})',
                        unsafe_allow_html=True,
                    )
