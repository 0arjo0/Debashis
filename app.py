import json
import re
import tempfile
import uuid
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pdfminer.high_level import extract_text


st.set_page_config(
    page_title="Travel Tale AI",
    layout="wide",
)

client = httpx.Client(verify=False)


SYSTEM_PROMPT = """
You are Travel Tale AI, an itinerary summarization and travel guidance assistant.

Your job:
1. Read the retrieved itinerary and travel-support context.
2. Create a concise, highly readable travel plan for the selected route and dates.
3. Highlight flights, stays, transfers, activities, risks, and timing dependencies.
4. Add practical local advice that feels personalized to the selected trip.

Rules:
- Use only the provided context. Do not invent reservations, booking IDs, prices, or times.
- If something is missing, say "Information not available".
- Keep the tone warm, calm, and travel-friendly.
- Prefer short sections and bullets.
- Mention date gaps, overlapping bookings, or tight connections when visible.
- If multiple itinerary options exist, focus on the one that best matches the chosen source, destination, and dates.

Output format:
## Trip Snapshot
- Route:
- Travel window:
- Main transport:
- Stay plan:

## Journey Summary
Write a short paragraph summarizing the trip in plain language.

## Important Connections
- ...

## Personalized Tips
- ...

## Missing Or Unclear Details
- ...
"""


def init_session_state() -> None:
    defaults = {
        "rag_ready": False,
        "vectorstore": None,
        "location_options": [],
        "uploaded_file_names": [],
        "data_preview": [],
        "last_summary": "",
        "vector_dir": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@400;600;700&family=Nunito:wght@400;600;700;800&display=swap');

        :root {
            --bg: linear-gradient(160deg, #fff5d6 0%, #d8f7ff 45%, #ffe0ea 100%);
            --card: rgba(255, 255, 255, 0.78);
            --card-strong: rgba(255, 255, 255, 0.92);
            --text: #22405b;
            --accent: #ff8c5a;
            --accent-2: #2bb6c9;
            --accent-3: #ffd166;
            --shadow: 0 18px 45px rgba(34, 64, 91, 0.12);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 209, 102, 0.35), transparent 25%),
                radial-gradient(circle at top right, rgba(43, 182, 201, 0.22), transparent 30%),
                radial-gradient(circle at bottom center, rgba(255, 140, 90, 0.20), transparent 28%),
                var(--bg);
            color: var(--text);
            font-family: "Nunito", sans-serif;
        }

        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3 {
            font-family: "Baloo 2", cursive !important;
            color: #12314a;
        }

        [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.72);
            backdrop-filter: blur(12px);
            border-left: 1px solid rgba(255,255,255,0.35);
        }

        .hero-shell {
            position: relative;
            overflow: hidden;
            background: linear-gradient(135deg, rgba(255,255,255,0.88), rgba(255,248,225,0.82));
            border: 1px solid rgba(255,255,255,0.55);
            border-radius: 28px;
            box-shadow: var(--shadow);
            padding: 2rem;
            margin-bottom: 1.5rem;
            animation: floatUp .7s ease-out;
        }

        .hero-shell::before,
        .hero-shell::after {
            content: "";
            position: absolute;
            border-radius: 999px;
            filter: blur(4px);
            opacity: 0.5;
        }

        .hero-shell::before {
            width: 180px;
            height: 180px;
            right: -35px;
            top: -55px;
            background: rgba(255, 209, 102, 0.5);
            animation: drift 7s infinite alternate ease-in-out;
        }

        .hero-shell::after {
            width: 140px;
            height: 140px;
            left: -25px;
            bottom: -35px;
            background: rgba(43, 182, 201, 0.28);
            animation: drift 6s infinite alternate-reverse ease-in-out;
        }

        .hero-title {
            font-size: 3rem;
            line-height: 1;
            margin: 0;
        }

        .hero-subtitle {
            font-size: 1.05rem;
            max-width: 760px;
            margin-top: .65rem;
            color: #35536f;
        }

        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: .65rem;
            margin-top: 1rem;
        }

        .pill {
            background: rgba(255,255,255,0.78);
            border: 1px solid rgba(18,49,74,0.07);
            border-radius: 999px;
            padding: .5rem .9rem;
            font-weight: 700;
            color: #285170;
        }

        .glass-card {
            background: var(--card);
            border: 1px solid rgba(255,255,255,0.52);
            backdrop-filter: blur(14px);
            border-radius: 24px;
            padding: 1.25rem 1.2rem;
            box-shadow: var(--shadow);
            animation: floatUp .8s ease-out;
        }

        .stat-card {
            background: linear-gradient(160deg, rgba(255,255,255,0.95), rgba(255,253,244,0.85));
            border: 1px solid rgba(18,49,74,0.06);
            border-radius: 22px;
            padding: 1rem;
            box-shadow: var(--shadow);
            min-height: 115px;
        }

        .stat-label {
            font-size: .85rem;
            text-transform: uppercase;
            letter-spacing: .08em;
            color: #58708a;
            margin-bottom: .4rem;
            font-weight: 800;
        }

        .stat-value {
            font-size: 1.45rem;
            font-weight: 800;
            color: #163650;
        }

        .section-tag {
            display: inline-block;
            padding: .35rem .7rem;
            border-radius: 999px;
            background: rgba(255, 140, 90, 0.12);
            color: #b95a2f;
            font-size: .8rem;
            font-weight: 800;
            letter-spacing: .04em;
            margin-bottom: .6rem;
        }

        .summary-box {
            background: var(--card-strong);
            border-radius: 24px;
            padding: 1.35rem;
            border: 1px solid rgba(18,49,74,0.06);
            box-shadow: var(--shadow);
        }

        .upload-note {
            padding: .9rem 1rem;
            border-radius: 18px;
            background: rgba(43, 182, 201, 0.09);
            color: #1a5674;
            font-weight: 700;
            border: 1px dashed rgba(43, 182, 201, 0.35);
        }

        div[data-testid="stFileUploader"] section {
            background: rgba(255,255,255,0.7);
            border: 2px dashed rgba(43,182,201,0.45);
            border-radius: 22px;
        }

        div[data-testid="stFileUploader"] small {
            color: #35536f;
        }

        .stButton > button, .stDownloadButton > button {
            border-radius: 999px;
            border: none;
            padding: .8rem 1.25rem;
            font-weight: 800;
            color: white;
            background: linear-gradient(135deg, #ff8c5a, #ffb347);
            box-shadow: 0 12px 25px rgba(255, 140, 90, 0.28);
            transition: transform .18s ease, box-shadow .18s ease;
        }

        .stButton > button:hover, .stDownloadButton > button:hover {
            transform: translateY(-1px) scale(1.01);
            box-shadow: 0 15px 28px rgba(255, 140, 90, 0.35);
        }

        @keyframes floatUp {
            from { opacity: 0; transform: translateY(14px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes drift {
            from { transform: translateY(0) translateX(0); }
            to { transform: translateY(12px) translateX(-12px); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize_text(raw: str) -> str:
    return re.sub(r"\s+", " ", raw).strip()


def flatten_json(data: Any, prefix: str = "") -> tuple[list[str], set[str]]:
    lines: list[str] = []
    locations: set[str] = set()
    location_keys = {"source", "destination", "origin", "from", "to", "city", "location", "airport"}

    if isinstance(data, dict):
        for key, value in data.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            nested_lines, nested_locations = flatten_json(value, new_prefix)
            lines.extend(nested_lines)
            locations.update(nested_locations)
            if str(key).lower() in location_keys and isinstance(value, str):
                locations.add(value.strip())
    elif isinstance(data, list):
        for index, item in enumerate(data):
            nested_lines, nested_locations = flatten_json(item, f"{prefix}[{index}]")
            lines.extend(nested_lines)
            locations.update(nested_locations)
    else:
        value = "" if data is None else str(data)
        label = prefix or "value"
        lines.append(f"{label}: {value}")

    return lines, {item for item in locations if item}


def extract_locations_from_dataframe(df: pd.DataFrame) -> set[str]:
    locations: set[str] = set()
    preferred_columns = {
        "source",
        "destination",
        "origin",
        "from",
        "to",
        "departure_city",
        "arrival_city",
        "city",
        "airport",
    }
    for column in df.columns:
        if str(column).strip().lower() in preferred_columns:
            values = df[column].dropna().astype(str).str.strip()
            locations.update(v for v in values if v)
    return locations


def parse_uploaded_file(uploaded_file) -> dict[str, Any]:
    suffix = Path(uploaded_file.name).suffix.lower()
    raw_text = ""
    locations: set[str] = set()
    preview = ""

    if suffix == ".pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_path = tmp.name
        raw_text = extract_text(temp_path) or ""
        preview = normalize_text(raw_text[:500])
        locations.update(re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+){0,2}\b", raw_text))

    elif suffix in {".json"}:
        data = json.loads(uploaded_file.getvalue().decode("utf-8"))
        lines, json_locations = flatten_json(data)
        raw_text = "\n".join(lines)
        locations.update(json_locations)
        preview = normalize_text(raw_text[:500])

    elif suffix in {".csv"}:
        df = pd.read_csv(uploaded_file)
        raw_text = df.to_csv(index=False)
        locations.update(extract_locations_from_dataframe(df))
        preview = df.head(6).to_string(index=False)

    elif suffix in {".txt", ".md"}:
        raw_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        preview = normalize_text(raw_text[:500])
        line_matches = re.findall(
            r"(?:source|destination|origin|from|to|city|airport)\s*[:=-]\s*([A-Za-z][A-Za-z\s-]{1,50})",
            raw_text,
            flags=re.IGNORECASE,
        )
        locations.update(match.strip() for match in line_matches if match.strip())

    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    cleaned_locations = sorted(
        {
            normalize_text(item)
            for item in locations
            if normalize_text(item)
            and len(normalize_text(item)) > 2
            and len(normalize_text(item)) < 60
        }
    )

    return {
        "name": uploaded_file.name,
        "text": raw_text.strip(),
        "locations": cleaned_locations,
        "preview": preview or "Preview not available",
    }


def build_vectorstore(file_payloads: list[dict[str, Any]]) -> Chroma:
    corpus = []
    for payload in file_payloads:
        header = f"FILE: {payload['name']}\n"
        corpus.append(header + payload["text"])

    combined_text = "\n\n".join(chunk for chunk in corpus if chunk.strip())
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=180)
    chunks = splitter.split_text(combined_text)

    if not chunks:
        raise ValueError("No readable itinerary content was found in the uploaded files.")

    embedding_model = OpenAIEmbeddings(
        base_url="https://genailab.tcs.in",
        model="azure/genailab-maas-text-embedding-3-large",
        api_key="sk-NSv6u2rCE44tzwp8WhMvLw",
        http_client=client,
    )

    vector_dir = Path(tempfile.gettempdir()) / f"travel_tale_{uuid.uuid4().hex}"
    st.session_state.vector_dir = str(vector_dir)
    vectordb = Chroma.from_texts(
        chunks,
        embedding_model,
        persist_directory=str(vector_dir),
    )
    vectordb.persist()
    return vectordb


def build_summary(
    source: str,
    destination: str,
    start_date: date,
    end_date: date,
    temperature: float,
    companion_type: str,
) -> str:
    llm = ChatOpenAI(
        base_url="https://genailab.tcs.in",
        model="azure/genailab-maas-gpt-35-turbo",
        api_key="sk-NSv6u2rCE44tzwp8WhMvLw",
        http_client=client,
        temperature=temperature,
    )

    query = (
        f"Summarize the travel itinerary for a trip from {source} to {destination}. "
        f"Travel start date: {start_date.strftime('%d/%m/%Y')}. "
        f"Return date: {end_date.strftime('%d/%m/%Y')}. "
        f"Traveler profile: {companion_type}. "
        "Prioritize transport, accommodation, activities, transfer timings, and local advice."
    )

    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 6})
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Selected route: {source} to {destination}\n"
                    f"Travel window: {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}\n"
                    f"Companion type: {companion_type}\n\n"
                    f"Retrieved itinerary context:\n{context}"
                )
            ),
        ]
    )
    return response.content


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-shell">
            <div class="section-tag">Smart Family Travel Planner</div>
            <h1 class="hero-title">Travel Tale AI</h1>
            <p class="hero-subtitle">
                Upload itinerary files, pick your route, and turn scattered bookings into a cheerful,
                crystal-clear travel story with timing alerts, connection highlights, and friendly local tips.
            </p>
            <div class="pill-row">
                <div class="pill">Multi-file upload</div>
                <div class="pill">Route-aware summaries</div>
                <div class="pill">Local advice</div>
                <div class="pill">Kid-friendly visual style</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stats(location_count: int, file_count: int) -> None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Uploaded Files</div>
                <div class="stat-value">{file_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Detected Places</div>
                <div class="stat-value">{location_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-label">Supported Formats</div>
                <div class="stat-value">PDF, CSV, JSON, TXT, MD</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    init_session_state()
    inject_styles()
    render_hero()

    with st.sidebar:
        st.markdown("### Tune the storyteller")
        temperature = st.slider(
            "Response creativity",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Lower values stay factual and tight. Higher values make the advice more expressive.",
        )
        companion_type = st.selectbox(
            "Traveler style",
            ["Family with kids", "Solo explorer", "Couple getaway", "Friends group", "Senior-friendly trip"],
        )
        st.markdown(
            """
            <div class="upload-note">
                Tip: if you do not have travel files yet, use the bundled synthetic sample in
                <code>sample_data/travel_itineraries.json</code>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    uploader_col, control_col = st.columns([1.15, 1])

    with uploader_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Upload travel data")
        st.caption("Add multiple files together for richer summaries. The app will combine them into one itinerary context.")
        uploaded_files = st.file_uploader(
            "Upload itinerary files",
            type=["pdf", "json", "csv", "txt", "md"],
            accept_multiple_files=True,
            help="Supported formats: PDF, JSON, CSV, TXT, and Markdown.",
        )

        process_clicked = st.button("Build Travel Knowledge Base", use_container_width=True)
        if process_clicked:
            if not uploaded_files:
                st.warning("Upload at least one itinerary or travel-support file to continue.")
            else:
                try:
                    parsed_payloads = [parse_uploaded_file(file) for file in uploaded_files]
                    vectorstore = build_vectorstore(parsed_payloads)
                    location_pool = sorted(
                        {
                            location
                            for payload in parsed_payloads
                            for location in payload["locations"]
                        }
                    )

                    st.session_state.vectorstore = vectorstore
                    st.session_state.rag_ready = True
                    st.session_state.location_options = location_pool
                    st.session_state.uploaded_file_names = [payload["name"] for payload in parsed_payloads]
                    st.session_state.data_preview = parsed_payloads
                    st.success("Travel files processed successfully. Pick your route and generate a summary.")
                except Exception as exc:
                    st.session_state.rag_ready = False
                    st.error(f"Could not process the uploaded files: {exc}")
        st.markdown("</div>", unsafe_allow_html=True)

    with control_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Trip inputs")
        if st.session_state.location_options:
            source = st.selectbox("Source", st.session_state.location_options, key="source_select")
            destination = st.selectbox("Destination", st.session_state.location_options, key="destination_select")
        else:
            st.info("Upload files first to unlock the route dropdowns.")
            source = st.selectbox("Source", ["Upload a file first"], disabled=True)
            destination = st.selectbox("Destination", ["Upload a file first"], disabled=True)

        start_date = st.date_input("Starting date", value=date.today(), format="DD/MM/YYYY")
        end_date = st.date_input("Returning date", value=date.today(), format="DD/MM/YYYY")

        generate_disabled = not st.session_state.rag_ready or not st.session_state.location_options
        generate_clicked = st.button("Generate Itinerary Summary", use_container_width=True, disabled=generate_disabled)
        st.markdown("</div>", unsafe_allow_html=True)

    render_stats(len(st.session_state.location_options), len(st.session_state.uploaded_file_names))

    if st.session_state.data_preview:
        st.markdown("### Uploaded travel context")
        preview_cols = st.columns(min(3, len(st.session_state.data_preview)))
        for index, payload in enumerate(st.session_state.data_preview[:3]):
            with preview_cols[index % len(preview_cols)]:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown(f"#### {payload['name']}")
                st.caption("Detected route values")
                if payload["locations"]:
                    st.write(", ".join(payload["locations"][:12]))
                else:
                    st.write("No structured route fields detected")
                st.caption("Preview")
                st.code(payload["preview"], language="text")
                st.markdown("</div>", unsafe_allow_html=True)

    if generate_clicked:
        if source == destination:
            st.error("Choose different source and destination locations.")
        elif end_date < start_date:
            st.error("Returning date must be the same as or later than the starting date.")
        else:
            with st.spinner("Crafting your travel story and practical trip advice..."):
                try:
                    st.session_state.last_summary = build_summary(
                        source=source,
                        destination=destination,
                        start_date=start_date,
                        end_date=end_date,
                        temperature=temperature,
                        companion_type=companion_type,
                    )
                except Exception as exc:
                    st.error(f"Summary generation failed: {exc}")

    if st.session_state.last_summary:
        st.markdown("### Your personalized itinerary")
        st.markdown('<div class="summary-box">', unsafe_allow_html=True)
        st.markdown(st.session_state.last_summary)
        st.markdown("</div>", unsafe_allow_html=True)
        st.download_button(
            "Download Summary",
            st.session_state.last_summary,
            file_name="travel_tale_summary.md",
            mime="text/markdown",
        )


if __name__ == "__main__":
    main()
