
from pathlib import Path

import pandas as pd
import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(
    page_title="AutoPulse AI",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 AutoPulse AI")
st.subheader("Connected Vehicle Intelligence Copilot")
st.caption("Local AI • Synthetic vehicle telemetry • RAG diagnostics")

DATA_PATH = Path("data/vehicle_telemetry.csv")
DB_PATH = Path("chroma_db")

@st.cache_resource
def load_resources():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = Chroma(
        collection_name="autopulse_manuals",
        persist_directory=str(DB_PATH),
        embedding_function=embeddings,
    )

    llm = ChatOllama(
        model="llama3.2:1b",
        temperature=0,
    )

    return vector_store, llm


if not DATA_PATH.exists():
    st.error("Telemetry CSV not found: data/vehicle_telemetry.csv")
    st.stop()

if not DB_PATH.exists():
    st.error("Vector database not found. Run: python ingest.py")
    st.stop()

try:
    vector_store, llm = load_resources()
except Exception as exc:
    st.error(f"Could not load local AI resources: {exc}")
    st.info("Ensure Ollama is running and both models are installed.")
    st.stop()

df = pd.read_csv(DATA_PATH)

if df.empty:
    st.error("The telemetry CSV contains no records.")
    st.stop()

if "vehicle_id" not in df.columns:
    st.error("The telemetry CSV must contain a vehicle_id column.")
    st.stop()

vehicle_ids = sorted(df["vehicle_id"].dropna().unique().tolist())

if not vehicle_ids:
    st.error("No vehicle IDs were found in the CSV.")
    st.stop()

selected_vehicle = st.sidebar.selectbox(
    "Select vehicle",
    vehicle_ids,
)

vehicle_df = df[df["vehicle_id"] == selected_vehicle].copy()

if "timestamp" in vehicle_df.columns:
    vehicle_df["timestamp"] = pd.to_datetime(
        vehicle_df["timestamp"], errors="coerce"
    )
    vehicle_df = vehicle_df.sort_values("timestamp")

latest = vehicle_df.iloc[-1]

st.markdown(f"### Vehicle: {selected_vehicle}")

c1, c2, c3 = st.columns(3)

if "battery_temp_c" in df.columns:
    c1.metric("Battery Temperature", f"{latest['battery_temp_c']} °C")
else:
    c1.metric("Battery Temperature", "Not available")

if "motor_temp_c" in df.columns:
    c2.metric("Motor Temperature", f"{latest['motor_temp_c']} °C")
else:
    c2.metric("Motor Temperature", "Not available")

dtc = str(latest.get("dtc", "NONE"))
c3.metric("Latest Diagnostic Code", dtc or "NONE")

st.caption(
    "All thresholds and vehicle records in this demo are synthetic, "
    "not OEM specifications or real vehicle diagnostics."
)

st.markdown("### Telemetry History")

chart_columns = [
    column
    for column in [
        "battery_temp_c",
        "motor_temp_c",
        "battery_soc_pct",
    ]
    if column in vehicle_df.columns
]

if chart_columns:
    st.line_chart(
        vehicle_df.set_index("timestamp")[chart_columns]
        if "timestamp" in vehicle_df.columns
        else vehicle_df[chart_columns]
    )
else:
    st.warning("No supported telemetry columns were found.")

with st.expander("View telemetry records"):
    st.dataframe(vehicle_df, use_container_width=True)

st.markdown("### Ask AutoPulse AI")

question = st.text_area(
    "Ask a question about the selected vehicle",
    placeholder=(
        "Example: What could cause a high battery temperature, "
        "and what should an engineer investigate?"
    ),
)

if st.button("Analyze", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching the manual and generating an answer..."):
            try:
                docs = vector_store.similarity_search(question, k=3)

                manual_context = "\n\n".join(
                    doc.page_content for doc in docs
                )

                telemetry_summary = vehicle_df.tail(5).to_string(
                    index=False
                )

                prompt = ChatPromptTemplate.from_template(
                    """
You are AutoPulse AI, a vehicle engineering assistant.

Use the supplied synthetic diagnostic manual and telemetry
to answer the user's question.

Rules:
- Clearly distinguish recorded demo data from general guidance.
- Do not invent diagnostic codes, measurements, or manual requirements.
- If the evidence is insufficient, say so.
- Recommend qualified engineering review for real vehicles.
- Never recommend directly changing vehicle controls.
- The supplied data and thresholds are synthetic demo examples.

Selected vehicle: {vehicle_id}

Recent telemetry:
{telemetry}

Retrieved manual excerpts:
{manual}

User question:
{question}

Provide:
1. Summary
2. Evidence from telemetry and manual
3. Suggested investigation steps
4. Limitations or missing information
"""
                )

                chain = prompt | llm

                response = chain.invoke(
                    {
                        "vehicle_id": selected_vehicle,
                        "telemetry": telemetry_summary,
                        "manual": manual_context or "No matching excerpts found.",
                        "question": question,
                    }
                )

                st.markdown("### AI Analysis")
                st.write(response.content)

                with st.expander("View retrieved manual evidence"):
                    if docs:
                        for index, doc in enumerate(docs, start=1):
                            st.markdown(f"**Excerpt {index}**")
                            st.write(doc.page_content)
                    else:
                        st.info("No matching manual excerpts were retrieved.")

            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                st.info(
                    "Check that Ollama is running and the required models "
                    "are available."
                )
