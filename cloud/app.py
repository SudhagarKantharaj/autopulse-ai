
from pathlib import Path
import re

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="AutoPulse AI",
    page_icon="🚗",
    layout="wide",
)

DATA_PATH = Path("data/vehicle_telemetry.csv")
MANUAL_PATH = Path("data/diagnostic_manual.txt")

BATTERY_LIMIT_C = 45.0
MOTOR_LIMIT_C = 90.0


@st.cache_data
def load_telemetry():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_manual():
    text = MANUAL_PATH.read_text(encoding="utf-8")
    paragraphs = [
        part.strip()
        for part in re.split(r"\n\s*\n", text)
        if part.strip()
    ]

    # Split long paragraphs into smaller searchable pieces.
    chunks = []
    for paragraph in paragraphs:
        if len(paragraph) <= 700:
            chunks.append(paragraph)
        else:
            chunks.extend(
                paragraph[i:i + 600]
                for i in range(0, len(paragraph), 500)
            )
    return chunks


def retrieve_manual(question, chunks, limit=3):
    """Simple keyword-overlap retrieval; no embeddings or API."""
    words = set(re.findall(r"[a-z0-9_]+", question.lower()))
    stop_words = {
        "the", "and", "for", "what", "with", "this", "that",
        "does", "could", "should", "from", "about", "vehicle",
        "please", "explain", "tell", "me",
    }
    words -= stop_words

    ranked = []
    for chunk in chunks:
        chunk_words = set(
            re.findall(r"[a-z0-9_]+", chunk.lower())
        )
        score = len(words & chunk_words)
        if score > 0:
            ranked.append((score, chunk))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in ranked[:limit]]


def interpret_vehicle(row):
    battery = float(row["battery_temp_c"])
    motor = float(row["motor_temp_c"])
    dtc = str(row.get("dtc", "NONE")).strip()

    findings = []

    if battery > BATTERY_LIMIT_C:
        findings.append(
            f"Battery temperature is {battery:.1f} °C, "
            f"above this demo's investigation threshold of "
            f"{BATTERY_LIMIT_C:.1f} °C."
        )
    else:
        findings.append(
            f"Battery temperature is {battery:.1f} °C, "
            f"not above this demo's threshold of "
            f"{BATTERY_LIMIT_C:.1f} °C."
        )

    if motor > MOTOR_LIMIT_C:
        findings.append(
            f"Motor temperature is {motor:.1f} °C, "
            f"above this demo's investigation threshold of "
            f"{MOTOR_LIMIT_C:.1f} °C."
        )
    else:
        findings.append(
            f"Motor temperature is {motor:.1f} °C, "
            f"not above this demo's threshold of "
            f"{MOTOR_LIMIT_C:.1f} °C."
        )

    if dtc in {"BATT_TEMP_HIGH", "MOTOR_TEMP_HIGH"}:
        findings.append(f"Recorded diagnostic code: {dtc}.")
    elif dtc and dtc.upper() != "NONE" and dtc.lower() != "nan":
        findings.append(f"Recorded diagnostic code: {dtc}.")
    else:
        findings.append(
            "No diagnostic code is recorded in this sample."
        )

    return findings


st.title("🚗 AutoPulse AI")
st.subheader("Connected Vehicle Intelligence Copilot")
st.caption(
    "Public cloud demo · Synthetic telemetry · "
    "Rule-based checks + lightweight manual retrieval"
)

if not DATA_PATH.exists() or not MANUAL_PATH.exists():
    st.error(
        "Required data files are missing. Check that both files "
        "are committed in the repository's data folder."
    )
    st.stop()

try:
    df = load_telemetry()
    manual_chunks = load_manual()
except Exception as exc:
    st.error(f"Could not load demo files: {exc}")
    st.stop()

required_columns = {
    "vehicle_id",
    "reading_number",
    "battery_soc_pct",
    "battery_temp_c",
    "motor_temp_c",
    "pack_voltage_v",
    "dtc",
}
missing = required_columns - set(df.columns)

if missing:
    st.error(f"CSV is missing columns: {', '.join(sorted(missing))}")
    st.stop()

vehicle_ids = sorted(df["vehicle_id"].dropna().unique().tolist())

if not vehicle_ids:
    st.error("No vehicle IDs found in the telemetry file.")
    st.stop()

selected_vehicle = st.sidebar.selectbox(
    "Select vehicle",
    vehicle_ids,
)

vehicle_df = df[df["vehicle_id"] == selected_vehicle].copy()
vehicle_df = vehicle_df.sort_values("reading_number")
latest = vehicle_df.iloc[-1]

battery = float(latest["battery_temp_c"])
motor = float(latest["motor_temp_c"])
soc = float(latest["battery_soc_pct"])
voltage = float(latest["pack_voltage_v"])

st.markdown(f"### Vehicle: {selected_vehicle}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Battery Temperature", f"{battery:.1f} °C")
c2.metric("Motor Temperature", f"{motor:.1f} °C")
c3.metric("Battery State of Charge", f"{soc:.1f}%")
c4.metric("Pack Voltage", f"{voltage:.1f} V")

flags = []
if battery > BATTERY_LIMIT_C:
    flags.append("Battery temperature exceeds the demo threshold.")
if motor > MOTOR_LIMIT_C:
    flags.append("Motor temperature exceeds the demo threshold.")

if flags:
    st.warning("Investigation flags: " + " ".join(flags))
else:
    st.success("No temperature threshold exceeded in the latest sample.")

st.caption(
    "All thresholds and readings are synthetic demonstration data, "
    "not OEM specifications or real vehicle diagnostics."
)

st.markdown("### Telemetry History")
st.line_chart(
    vehicle_df.set_index("reading_number")[
        ["battery_temp_c", "motor_temp_c", "battery_soc_pct"]
    ]
)

with st.expander("View telemetry records"):
    st.dataframe(vehicle_df, use_container_width=True)

st.markdown("### Ask AutoPulse AI")
question = st.text_area(
    "Enter an engineering question",
    placeholder=(
        "Example: What should an engineer investigate if "
        "battery temperature is high?"
    ),
)

if st.button("Analyze", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        st.markdown("### Diagnostic Summary")

        for finding in interpret_vehicle(latest):
            st.write(f"- {finding}")

        st.markdown("### Suggested Investigation")
        if battery > BATTERY_LIMIT_C:
            st.write(
                "- Review battery temperature trends and related "
                "diagnostic codes."
            )
            st.write(
                "- Review relevant cooling-system and operating "
                "information if available."
            )
        if motor > MOTOR_LIMIT_C:
            st.write(
                "- Review motor temperature trends and the recorded "
                "diagnostic code."
            )
        if not flags:
            st.write(
                "- No temperature flag was triggered by the current "
                "demo rules. Review the trend and any recorded DTC "
                "if investigating a concern."
            )

        st.write(
            "- Compare observations with approved, vehicle-specific "
            "service documentation before making a real-world decision."
        )

        evidence = retrieve_manual(question, manual_chunks)

        st.markdown("### Retrieved Manual Evidence")
        if evidence:
            for i, excerpt in enumerate(evidence, start=1):
                with st.expander(f"Evidence excerpt {i}"):
                    st.write(excerpt)
        else:
            st.info(
                "No matching manual passage was found by keyword search. "
                "The diagnostic summary above is based on demo rules "
                "and telemetry, not a matching manual excerpt."
            )

        st.caption(
            "This cloud version uses deterministic rules and keyword "
            "retrieval, not an LLM. It is a prototype, not an "
            "OEM-approved diagnostic system."
        )
