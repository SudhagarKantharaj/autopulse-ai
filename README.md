# AutoPulse AI — Connected Vehicle Intelligence Copilot

AutoPulse AI is an experimental automotive AI prototype exploring how Retrieval-Augmented Generation (RAG) can connect vehicle telemetry with engineering documentation to support diagnostic investigations.

## Problem

Connected vehicle teams work with telemetry, diagnostic trouble codes, and technical manuals. Finding relevant evidence across these sources can take time and require manual investigation.

## Solution

AutoPulse AI combines:

- Synthetic connected-vehicle telemetry
- Engineering-manual retrieval using RAG
- LangChain orchestration
- Chroma vector database
- Ollama local language models and embeddings
- Streamlit dashboard

## Current capabilities

- Vehicle selection and telemetry visualization
- Retrieval of relevant diagnostic-manual excerpts
- AI-generated explanations and suggested investigation steps
- Local model execution without a paid OpenAI API key

## Technology stack

Python · LangChain · Ollama · Chroma · Streamlit · Pandas

## Roadmap

- Deterministic telemetry anomaly detection
- Diagnostic trouble code intelligence
- Evidence-grounded answers with clear uncertainty handling
- Fleet-level anomaly comparison
- Human-in-the-loop investigation workflows

## Running locally

1. Install Python, Git, and Ollama.
2. Install dependencies from `requirements.txt`.
3. Download the `llama3.2:1b` and `nomic-embed-text` models using Ollama.
4. Generate or verify the synthetic data files.
5. Run `python ingest.py`.
6. Run `python -m streamlit run app.py`.

## Limitations

This is a learning and product-validation prototype using synthetic data. It is not an OEM-approved diagnostic system, does not access live vehicle data, and must not be used to make real-world vehicle safety or maintenance decisions.

## Author

Built as an exploration of evidence-grounded AI for connected vehicles and automotive engineering.
