"""
GuardianNodes RoadSoS — Advanced Emergency Response Dashboard
=============================================================
Production-grade Streamlit application integrating:
  1. IntelligentTriageEngine  — ZSC AI classification + urgency scoring
  2. GraphRoutingEngine       — NetworkX Dijkstra traffic-aware routing
  3. MeshProtocolSimulator    — BLE/Wi-Fi Direct packet relay simulation
  4. Persistent session state metrics dashboard
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from modules.triage_engine import IntelligentTriageEngine, CANDIDATE_LABELS
from modules.routing_engine import GraphRoutingEngine, VEHICLE_PROFILES
from modules.mesh_simulator import MeshProtocolSimulator

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("roadsaos.app")

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GuardianNodes RoadSoS",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark glassmorphism premium UI
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary:    #0a0e1a;
    --bg-secondary:  #0f1629;
    --bg-card:       rgba(15, 22, 41, 0.85);
    --glass-border:  rgba(255, 255, 255, 0.08);
    --accent-red:    #ff4757;
    --accent-orange: #ff6b35;
    --accent-teal:   #00d2ff;
    --accent-green:  #2ed573;
    --accent-purple: #a855f7;
    --accent-yellow: #ffd32a;
    --text-primary:  #e8eaf6;
    --text-secondary:#8892b0;
    --text-dim:      #4a5568;
    --gradient-hero: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0d1b2a 100%);
}

* { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Main container */
.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* === HERO BANNER === */
.hero-banner {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b35 40%, #0a1628 100%);
    border-bottom: 1px solid rgba(0, 210, 255, 0.15);
    padding: 1.5rem 2.5rem 1rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -20%;
    width: 60%;
    height: 200%;
    background: radial-gradient(ellipse, rgba(255, 71, 87, 0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    top: -30%;
    right: -10%;
    width: 40%;
    height: 160%;
    background: radial-gradient(ellipse, rgba(0, 210, 255, 0.05) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 900;
    letter-spacing: -0.03em;
    background: linear-gradient(90deg, #ff4757, #ff6b35, #ffd32a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
}
.hero-subtitle {
    font-size: 0.85rem;
    color: var(--text-secondary);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.3rem;
    font-weight: 500;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(46, 213, 115, 0.1);
    border: 1px solid rgba(46, 213, 115, 0.3);
    color: #2ed573;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-left: 1rem;
}
.hero-badge::before {
    content: '';
    width: 6px; height: 6px;
    background: #2ed573;
    border-radius: 50%;
    animation: pulse-dot 1.5s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(1.4); }
}

/* === GLASS CARD === */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 1.4rem;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
    margin-bottom: 1rem;
}
.glass-card:hover {
    border-color: rgba(0, 210, 255, 0.2);
    box-shadow: 0 4px 32px rgba(0,0,0,0.5), 0 0 0 1px rgba(0,210,255,0.08);
}
.card-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.card-title::before {
    content: '';
    display: inline-block;
    width: 3px;
    height: 14px;
    border-radius: 2px;
    background: var(--accent-teal);
}

/* === URGENCY METER === */
.urgency-meter-container { margin: 0.8rem 0; }
.urgency-value {
    font-size: 3.5rem;
    font-weight: 900;
    line-height: 1;
    letter-spacing: -0.04em;
}
.urgency-label {
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 600;
    margin-top: 2px;
}
.urgency-bar-track {
    height: 8px;
    background: rgba(255,255,255,0.06);
    border-radius: 4px;
    overflow: hidden;
    margin-top: 0.6rem;
}
.urgency-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}

/* === SCORE BARS (label breakdown) === */
.score-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
    font-size: 0.78rem;
}
.score-label { flex: 0 0 200px; color: var(--text-secondary); }
.score-track {
    flex: 1;
    height: 5px;
    background: rgba(255,255,255,0.06);
    border-radius: 3px;
    overflow: hidden;
}
.score-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.6s ease;
}
.score-pct {
    flex: 0 0 40px;
    text-align: right;
    font-size: 0.72rem;
    color: var(--text-secondary);
    font-family: 'JetBrains Mono', monospace;
}

/* === METRICS ROW === */
.metric-card {
    background: rgba(15, 22, 41, 0.9);
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1;
}
.metric-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-secondary);
    margin-top: 4px;
    font-weight: 600;
}
.metric-delta {
    font-size: 0.7rem;
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
}

/* === HOP LOG === */
.hop-log-entry {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 8px 10px;
    border-radius: 8px;
    margin-bottom: 4px;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    border-left: 3px solid;
    background: rgba(0,0,0,0.2);
}
.hop-pass    { border-color: #2ed573; color: #2ed573; }
.hop-retrans { border-color: #ffd32a; color: #ffd32a; }
.hop-corrupt { border-color: #ff4757; color: #ff4757; }
.hop-lost    { border-color: #8892b0; color: #8892b0; }

/* === ROUTE SEGMENT === */
.route-seg {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 10px;
    border-radius: 6px;
    margin-bottom: 3px;
    font-size: 0.76rem;
    background: rgba(0,0,0,0.15);
}
.seg-highway   { border-left: 3px solid #00d2ff; }
.seg-arterial  { border-left: 3px solid #a855f7; }
.seg-local     { border-left: 3px solid #ffd32a; }
.seg-service   { border-left: 3px solid #ff4757; }

/* === TAGS === */
.entity-tag {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(168, 85, 247, 0.1);
    border: 1px solid rgba(168, 85, 247, 0.3);
    color: #c084fc;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 20px;
    margin: 2px;
    font-family: 'JetBrains Mono', monospace;
}

/* === UNIT BADGE === */
.unit-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0, 210, 255, 0.08);
    border: 1px solid rgba(0, 210, 255, 0.2);
    color: var(--accent-teal);
    font-size: 0.75rem;
    padding: 4px 12px;
    border-radius: 8px;
    margin: 3px;
    font-weight: 600;
}

/* === PACKET HEX DUMP === */
.hex-dump {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #64748b;
    background: rgba(0,0,0,0.3);
    border-radius: 8px;
    padding: 10px 12px;
    word-break: break-all;
    line-height: 1.8;
    border: 1px solid rgba(255,255,255,0.04);
}
.hex-dump span { color: #00d2ff; }

/* === SECTION DIVIDER === */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,210,255,0.15), transparent);
    margin: 1.5rem 0;
}

/* === ALERT BANNERS === */
.alert-critical {
    background: rgba(255, 71, 87, 0.08);
    border: 1px solid rgba(255, 71, 87, 0.3);
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 0.8rem;
    color: #ff6b7a;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0.8rem;
}
.alert-success {
    background: rgba(46, 213, 115, 0.08);
    border: 1px solid rgba(46, 213, 115, 0.3);
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 0.8rem;
    color: #5adf8c;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0.8rem;
}
.alert-warn {
    background: rgba(255, 211, 42, 0.08);
    border: 1px solid rgba(255, 211, 42, 0.3);
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 0.8rem;
    color: #ffd32a;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0.8rem;
}

/* Streamlit component overrides */
.stTextArea textarea {
    background: rgba(10, 14, 26, 0.8) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
}
.stTextArea textarea:focus {
    border-color: rgba(0, 210, 255, 0.4) !important;
    box-shadow: 0 0 0 2px rgba(0, 210, 255, 0.1) !important;
}
.stSelectbox > div > div {
    background: rgba(10, 14, 26, 0.8) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #ff4757 0%, #ff6b35 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.5rem !important;
    font-size: 0.85rem !important;
    transition: all 0.2s ease !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(255, 71, 87, 0.4) !important;
}
div[data-testid="stMetric"] {
    background: rgba(15, 22, 41, 0.8) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    padding: 0.8rem 1rem !important;
}
div[data-testid="stMetric"] label {
    color: var(--text-secondary) !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
div[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-weight: 800 !important;
}
/* Chart backgrounds */
.js-plotly-plot .plotly, .js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}
/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,210,255,0.3); }

/* Content padding */
.content-wrapper { padding: 1.2rem 2rem; }

/* Stagger animation */
@keyframes fade-up {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.animate-fade-up { animation: fade-up 0.4s ease both; }
</style>
"""

# ---------------------------------------------------------------------------
# Demo scenarios for auto-initialization
# ---------------------------------------------------------------------------
DEMO_SCENARIOS = [
    {
        "text":     "Major explosion at the central fuel depot! Three vehicles are engulfed in flames. Two casualties visible near the blast zone. Heavy smoke covering the highway.",
        "lat":      28.612,
        "lon":      77.213,
        "vehicle":  "fire_engine",
    },
    {
        "text":     "Head-on collision on the expressway. Five cars involved. At least four people critically injured and one person trapped under a truck. Need immediate help.",
        "lat":      28.619,
        "lon":      77.228,
        "vehicle":  "ambulance",
    },
    {
        "text":     "Minor fender bender near the shopping mall. Two cars, no injuries reported. Traffic is backing up slightly.",
        "lat":      28.607,
        "lon":      77.209,
        "vehicle":  "police_car",
    },
    {
        "text":     "Chemical tanker overturned on NH-48. Toxic liquid spilling onto the road. Fumes are spreading. Evacuate the area immediately!",
        "lat":      28.625,
        "lon":      77.235,
        "vehicle":  "hazmat_unit",
    },
]

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    if "dispatch_history" not in st.session_state:
        st.session_state.dispatch_history = pd.DataFrame(columns=[
            "timestamp", "incident_type", "urgency_index",
            "vehicle_dispatched", "route_time_min",
            "mesh_delivery_ratio", "mesh_hops", "lat", "lon",
        ])
    if "engines_ready" not in st.session_state:
        st.session_state.engines_ready = False
    if "triage_engine" not in st.session_state:
        st.session_state.triage_engine = None
    if "routing_engine" not in st.session_state:
        st.session_state.routing_engine = None
    if "mesh_simulator" not in st.session_state:
        st.session_state.mesh_simulator = None
    if "last_triage" not in st.session_state:
        st.session_state.last_triage = None
    if "last_route" not in st.session_state:
        st.session_state.last_route = None
    if "last_mesh" not in st.session_state:
        st.session_state.last_mesh = None
    if "demo_run" not in st.session_state:
        st.session_state.demo_run = False
    if "dispatch_count" not in st.session_state:
        st.session_state.dispatch_count = 0
    # Prefill state for widgets (never write to widget keys directly after render)
    if "_prefill_text" not in st.session_state:
        st.session_state._prefill_text = DEMO_SCENARIOS[0]["text"]
    if "_prefill_lat" not in st.session_state:
        st.session_state._prefill_lat = DEMO_SCENARIOS[0]["lat"]
    if "_prefill_lon" not in st.session_state:
        st.session_state._prefill_lon = DEMO_SCENARIOS[0]["lon"]
    if "_prefill_vehicle" not in st.session_state:
        st.session_state._prefill_vehicle = DEMO_SCENARIOS[0]["vehicle"]


@st.cache_resource(show_spinner=False)
def load_engines():
    """Load all three engines (cached across sessions)."""
    triage   = IntelligentTriageEngine()
    routing  = GraphRoutingEngine(seed=42)
    mesh     = MeshProtocolSimulator(seed=None)
    return triage, routing, mesh


# ---------------------------------------------------------------------------
# Helper renderers
# ---------------------------------------------------------------------------

def urgency_color(idx: float) -> str:
    if idx >= 8.0:
        return "#ff4757"
    elif idx >= 6.0:
        return "#ff6b35"
    elif idx >= 4.0:
        return "#ffd32a"
    else:
        return "#2ed573"


def render_urgency_gauge(urgency: float) -> str:
    color = urgency_color(urgency)
    pct   = (urgency - 1.0) / 9.0 * 100
    label = (
        "CRITICAL" if urgency >= 8.0
        else "HIGH" if urgency >= 6.0
        else "MODERATE" if urgency >= 4.0
        else "LOW"
    )
    return f"""
    <div class="urgency-meter-container">
        <div class="urgency-value" style="color:{color}">{urgency:.2f}</div>
        <div class="urgency-label" style="color:{color}">{label} URGENCY</div>
        <div class="urgency-bar-track">
            <div class="urgency-bar-fill" style="width:{pct:.1f}%;
                 background:linear-gradient(90deg, {color}88, {color});"></div>
        </div>
    </div>
    """


SCORE_COLORS = ["#ff4757", "#ff6b35", "#ffd32a", "#2ed573", "#00d2ff", "#a855f7"]


def render_score_bars(label_scores: dict[str, float]) -> str:
    sorted_scores = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
    bars = ""
    for i, (label, score) in enumerate(sorted_scores):
        color = SCORE_COLORS[i % len(SCORE_COLORS)]
        pct   = score * 100
        bars += f"""
        <div class="score-row">
            <div class="score-label">{label}</div>
            <div class="score-track">
                <div class="score-fill" style="width:{pct:.1f}%;background:{color};"></div>
            </div>
            <div class="score-pct">{pct:.1f}%</div>
        </div>"""
    return bars


def render_entities(entities: dict[str, list[str]]) -> str:
    if not entities:
        return "<span style='color:#4a5568;font-size:0.78rem;'>No specific entities extracted.</span>"
    tags = ""
    icon_map = {
        "vehicle_count":  "🚗",
        "casualty_count": "🩺",
        "fire_mention":   "🔥",
        "hazmat_mention": "☣️",
        "speed_mention":  "⚡",
    }
    for entity_type, vals in entities.items():
        icon = icon_map.get(entity_type, "📌")
        label = entity_type.replace("_", " ").title()
        for v in vals:
            tags += f'<span class="entity-tag">{icon} {label}: {v}</span>'
    return tags


def render_hop_log(hop_logs) -> str:
    if not hop_logs:
        return ""
    html = ""
    for hop in hop_logs:
        status_class = {
            "PASS":           "hop-pass",
            "RETRANSMIT_OK":  "hop-retrans",
            "CORRUPTED":      "hop-corrupt",
            "LOST":           "hop-lost",
            "RETRANSMIT_FAIL":"hop-lost",
        }.get(hop.status, "hop-lost")

        status_icon = {
            "PASS":           "✓",
            "RETRANSMIT_OK":  "↻",
            "CORRUPTED":      "⚠",
            "LOST":           "✗",
            "RETRANSMIT_FAIL":"✗",
        }.get(hop.status, "?")

        crc_info = (
            f"CRC: {hop.crc_received}"
            if hop.crc_expected == hop.crc_received
            else f"CRC ERR {hop.crc_expected}≠{hop.crc_received}"
        )

        html += f"""
        <div class="hop-log-entry {status_class}">
            <span>{status_icon}</span>
            <span>HOP {hop.hop_number:02d} | {hop.relay_node_id} | {hop.latency_ms:.1f}ms | {hop.rssi_dbm} dBm | {crc_info}</span>
        </div>"""
    return html


def render_route_segments(segments: list[dict], max_show: int = 8) -> str:
    if not segments:
        return ""
    html = ""
    shown = segments[:max_show]
    type_class = {
        "highway":     "seg-highway",
        "arterial":    "seg-arterial",
        "local":       "seg-local",
        "service_lane":"seg-service",
    }
    for seg in shown:
        cls    = type_class.get(seg["road_type"], "seg-local")
        t_icon = {"clear": "🟢", "moderate": "🟡", "heavy": "🔴", "gridlock": "⛔"}.get(seg["traffic_level"], "⚪")
        html += f"""
        <div class="route-seg {cls}">
            <span>{seg['from']}→{seg['to']} | {seg['road_type'].replace('_',' ').title()}</span>
            <span>{t_icon} {seg['traffic_level'].title()} | {seg['distance_km']:.3f} km</span>
        </div>"""
    if len(segments) > max_show:
        html += f"<div style='color:#4a5568;font-size:0.72rem;text-align:center;margin-top:4px;'>+{len(segments)-max_show} more segments…</div>"
    return html


def render_hex_dump(hex_str: str) -> str:
    # Split into groups of 2 for readability (bytes)
    chunks = [hex_str[i:i+2] for i in range(0, len(hex_str), 2)]
    # Colour-code: payload bytes vs CRC bytes (last 4 bytes = 8 chars)
    payload_bytes = chunks[:-4]
    crc_bytes     = chunks[-4:]
    payload_html  = " ".join(payload_bytes)
    crc_html      = " ".join(f'<span>{b}</span>' for b in crc_bytes)
    return f'<div class="hex-dump">{payload_html} <span style="color:#ff4757">{crc_html}</span></div>'


# ---------------------------------------------------------------------------
# Plotly chart helpers
# ---------------------------------------------------------------------------

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#8892b0", size=11),
    margin=dict(l=10, r=10, t=30, b=10),
    showlegend=False,
)


def make_urgency_radar(label_scores: dict[str, float]) -> go.Figure:
    labels = list(label_scores.keys())
    values = list(label_scores.values())
    values.append(values[0])
    labels.append(labels[0])
    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=labels,
        fill="toself",
        fillcolor="rgba(0, 210, 255, 0.12)",
        line=dict(color="#00d2ff", width=2),
        marker=dict(size=5, color="#00d2ff"),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showticklabels=False,
                gridcolor="rgba(255,255,255,0.06)",
                linecolor="rgba(255,255,255,0.04)",
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.06)",
                linecolor="rgba(255,255,255,0.04)",
                tickfont=dict(size=10, color="#8892b0"),
            ),
        ),
        height=260,
    )
    return fig


def make_route_map(
    all_coords: dict[int, tuple[float, float]],
    path_coords: list[tuple[float, float]],
    source_node: int,
    target_node: int,
    src_coords: tuple[float, float],
    tgt_coords: tuple[float, float],
) -> go.Figure:
    """Create a Plotly scatter geo showing the road network and optimal route."""
    all_lats = [c[0] for c in all_coords.values()]
    all_lons = [c[1] for c in all_coords.values()]

    path_lats = [c[0] for c in path_coords]
    path_lons = [c[1] for c in path_coords]

    fig = go.Figure()

    # All nodes (background)
    fig.add_trace(go.Scattermapbox(
        lat=all_lats,
        lon=all_lons,
        mode="markers",
        marker=dict(size=4, color="rgba(136,146,176,0.25)"),
        name="Road Network",
        hoverinfo="skip",
    ))

    # Route path
    fig.add_trace(go.Scattermapbox(
        lat=path_lats,
        lon=path_lons,
        mode="lines+markers",
        line=dict(width=4, color="#00d2ff"),
        marker=dict(size=6, color="#00d2ff"),
        name="Optimal Route",
    ))

    # Source
    fig.add_trace(go.Scattermapbox(
        lat=[src_coords[0]],
        lon=[src_coords[1]],
        mode="markers+text",
        marker=dict(size=14, color="#2ed573", symbol="circle"),
        text=["🏥 Depot"],
        textposition="top right",
        textfont=dict(size=11, color="#2ed573"),
        name="Depot",
    ))

    # Target (incident)
    fig.add_trace(go.Scattermapbox(
        lat=[tgt_coords[0]],
        lon=[tgt_coords[1]],
        mode="markers+text",
        marker=dict(size=16, color="#ff4757", symbol="circle"),
        text=["🚨 Incident"],
        textposition="top right",
        textfont=dict(size=11, color="#ff4757"),
        name="Incident",
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=28.614, lon=77.222),
            zoom=12.5,
        ),
        height=340,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(10,14,26,0.8)",
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1,
            font=dict(size=10, color="#8892b0"),
        ),
    )
    return fig


def make_response_time_chart(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history["timestamp"],
        y=history["route_time_min"],
        mode="lines+markers",
        line=dict(color="#00d2ff", width=2.5, shape="spline"),
        marker=dict(size=6, color="#00d2ff",
                    line=dict(color="rgba(0,210,255,0.3)", width=6)),
        fill="tozeroy",
        fillcolor="rgba(0,210,255,0.06)",
        name="Response Time (min)",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Est. Response Time per Dispatch (min)", font=dict(size=12, color="#8892b0")),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False,
                   tickfont=dict(size=10)),
        height=220,
    )
    return fig


def make_delivery_chart(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = [
        "#2ed573" if r >= 0.75 else "#ffd32a" if r >= 0.5 else "#ff4757"
        for r in history["mesh_delivery_ratio"]
    ]
    fig.add_trace(go.Bar(
        x=list(range(1, len(history) + 1)),
        y=history["mesh_delivery_ratio"] * 100,
        marker=dict(color=colors, line=dict(width=0)),
        name="Delivery %",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Mesh Packet Delivery Rate (%)", font=dict(size=12, color="#8892b0")),
        xaxis=dict(title="Dispatch #", showgrid=False, zeroline=False,
                   tickfont=dict(size=10)),
        yaxis=dict(range=[0, 105], gridcolor="rgba(255,255,255,0.04)",
                   ticksuffix="%", tickfont=dict(size=10)),
        height=220,
    )
    return fig


def make_urgency_trend_chart(history: pd.DataFrame) -> go.Figure:
    colors = [urgency_color(u) for u in history["urgency_index"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(history) + 1)),
        y=history["urgency_index"],
        mode="lines+markers",
        line=dict(color="#a855f7", width=2, shape="spline"),
        marker=dict(size=8, color=colors,
                    line=dict(color="rgba(168,85,247,0.3)", width=5)),
        fill="tozeroy",
        fillcolor="rgba(168,85,247,0.05)",
    ))
    fig.add_hline(y=8.0, line=dict(color="#ff4757", width=1, dash="dot"),
                  annotation_text="Critical threshold",
                  annotation_font=dict(size=9, color="#ff4757"))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Urgency Index per Dispatch", font=dict(size=12, color="#8892b0")),
        xaxis=dict(title="Dispatch #", showgrid=False, zeroline=False,
                   tickfont=dict(size=10)),
        yaxis=dict(range=[0, 10.5], gridcolor="rgba(255,255,255,0.04)",
                   tickfont=dict(size=10)),
        height=220,
    )
    return fig


# ---------------------------------------------------------------------------
# Core dispatch logic
# ---------------------------------------------------------------------------

def run_full_dispatch(
    witness_text: str,
    lat: float,
    lon: float,
    vehicle_type: str,
    engines: tuple,
) -> None:
    """Run the complete pipeline: triage → routing → mesh → log."""
    triage_engine, routing_engine, mesh_sim = engines

    # ── 1. AI Triage ──────────────────────────────────────────────────
    with st.spinner("🧠 Running AI triage classification …"):
        try:
            triage = triage_engine.analyze(witness_text)
            st.session_state.last_triage = triage
        except Exception as exc:
            st.error(f"Triage failed: {exc}")
            return

    # ── 2. Graph Routing ──────────────────────────────────────────────
    with st.spinner("🗺️ Computing traffic-aware route …"):
        routing_engine.refresh_traffic()
        target_node = random.randint(50, 95)
        route = routing_engine.route(
            source_node=0,
            target_node=target_node,
            vehicle_type=vehicle_type,
        )
        st.session_state.last_route = route

    # ── 3. Mesh Transmission ──────────────────────────────────────────
    with st.spinner("📡 Simulating BLE mesh relay …"):
        mesh_result = mesh_sim.transmit_sync(
            latitude=lat,
            longitude=lon,
            urgency=triage.urgency_index,
        )
        st.session_state.last_mesh = mesh_result

    # ── 4. Log to history ─────────────────────────────────────────────
    new_row = {
        "timestamp":          datetime.datetime.now().strftime("%H:%M:%S"),
        "incident_type":      triage.top_label,
        "urgency_index":      triage.urgency_index,
        "vehicle_dispatched": VEHICLE_PROFILES[vehicle_type]["label"],
        "route_time_min":     route.estimated_time_min,
        "mesh_delivery_ratio":mesh_result.delivery_ratio,
        "mesh_hops":          mesh_result.total_hops,
        "lat":                lat,
        "lon":                lon,
    }
    st.session_state.dispatch_history = pd.concat(
        [st.session_state.dispatch_history, pd.DataFrame([new_row])],
        ignore_index=True,
    )
    st.session_state.dispatch_count += 1


# ---------------------------------------------------------------------------
# Panel renderers
# ---------------------------------------------------------------------------

def render_triage_panel(triage) -> None:
    st.markdown('<div class="card-title">🧠 AI Triage Analysis</div>', unsafe_allow_html=True)

    # Urgency gauge
    st.markdown(render_urgency_gauge(triage.urgency_index), unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    # Top classification
    color = urgency_color(triage.urgency_index)
    st.markdown(
        f"<div style='font-size:1rem;font-weight:700;color:{color};margin-bottom:0.4rem;'>"
        f"⚡ {triage.top_label}</div>",
        unsafe_allow_html=True,
    )

    # Radar chart
    st.plotly_chart(
        make_urgency_radar(triage.label_scores),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # Score bars
    st.markdown('<div class="card-title" style="margin-top:0.5rem;">Classification Confidence</div>', unsafe_allow_html=True)
    st.markdown(render_score_bars(triage.label_scores), unsafe_allow_html=True)

    # Entities
    st.markdown('<div class="card-title" style="margin-top:0.8rem;">🔍 Extracted Entities</div>', unsafe_allow_html=True)
    st.markdown(render_entities(triage.entities), unsafe_allow_html=True)

    # Recommended units
    st.markdown('<div class="card-title" style="margin-top:0.8rem;">🚨 Recommended Dispatch Units</div>', unsafe_allow_html=True)
    units_html = "".join(f'<span class="unit-badge">{u}</span>' for u in triage.recommended_units)
    st.markdown(units_html, unsafe_allow_html=True)


def render_routing_panel(route, routing_engine) -> None:
    st.markdown('<div class="card-title">🗺️ Traffic-Aware Route</div>', unsafe_allow_html=True)

    profile = VEHICLE_PROFILES[route.vehicle_type]
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Vehicle", profile["label"])
    with col_b:
        st.metric("Est. Time", f"{route.estimated_time_min:.1f} min")
    with col_c:
        st.metric("Route Hops", len(route.path))

    st.metric("Total Route Cost", f"{route.total_cost:.3f} units")

    if route.congestion_zones:
        st.markdown(
            f'<div class="alert-warn">⚠ {len(route.congestion_zones)} congestion zone(s) detected on route. '
            f'Emergency override corridor activated.</div>',
            unsafe_allow_html=True,
        )

    # Map
    all_coords = routing_engine.all_node_coords
    src_coords = routing_engine.node_coords(route.source_node)
    tgt_coords = routing_engine.node_coords(route.target_node)

    try:
        fig = make_route_map(
            all_coords, route.path_coords,
            route.source_node, route.target_node,
            src_coords, tgt_coords,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except Exception:
        # Fallback: simple lat/lon scatter if mapbox fails
        lats = [c[0] for c in route.path_coords]
        lons = [c[1] for c in route.path_coords]
        fig_fallback = go.Figure(go.Scatter(
            x=lons, y=lats, mode="lines+markers",
            line=dict(color="#00d2ff", width=3),
            marker=dict(size=6, color="#00d2ff"),
        ))
        fig_fallback.update_layout(**PLOTLY_LAYOUT, height=300,
                                   xaxis_title="Longitude", yaxis_title="Latitude")
        st.plotly_chart(fig_fallback, use_container_width=True, config={"displayModeBar": False})

    # Road segments
    st.markdown('<div class="card-title" style="margin-top:0.8rem;">Road Segments</div>', unsafe_allow_html=True)
    st.markdown(render_route_segments(route.road_segments), unsafe_allow_html=True)


def render_mesh_panel(mesh_result) -> None:
    st.markdown('<div class="card-title">📡 Mesh Relay Protocol Log</div>', unsafe_allow_html=True)

    # Status banner
    if mesh_result.final_status == "DELIVERED":
        st.markdown(
            f'<div class="alert-success">✓ PACKET DELIVERED — {mesh_result.hops_completed}/{mesh_result.total_hops} hops '
            f'| Delivery ratio: {mesh_result.delivery_ratio*100:.1f}% '
            f'| Total latency: {mesh_result.total_latency_ms:.1f} ms</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="alert-critical">✗ DELIVERY FAILED — {mesh_result.hops_completed}/{mesh_result.total_hops} hops '
            f'| Delivery ratio: {mesh_result.delivery_ratio*100:.1f}% '
            f'| Fallback: GPRS uplink triggered</div>',
            unsafe_allow_html=True,
        )

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Total Hops", mesh_result.total_hops)
        st.metric("Successful Hops", mesh_result.hops_completed)
    with col_b:
        st.metric("Delivery Rate", f"{mesh_result.delivery_ratio*100:.1f}%")
        st.metric("Total Latency", f"{mesh_result.total_latency_ms:.1f} ms")

    # Raw payload hex dump
    st.markdown('<div class="card-title" style="margin-top:0.8rem;">📦 Binary Payload (24 bytes — CRC-32 in red)</div>', unsafe_allow_html=True)
    st.markdown(render_hex_dump(mesh_result.raw_payload_hex), unsafe_allow_html=True)

    # Hop log
    st.markdown('<div class="card-title" style="margin-top:0.8rem;">Hop-by-Hop Transmission Log</div>', unsafe_allow_html=True)
    st.markdown(render_hop_log(mesh_result.hop_logs), unsafe_allow_html=True)


def render_metrics_dashboard(history: pd.DataFrame) -> None:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-title" style="font-size:0.75rem;margin-bottom:1.2rem;">'
        '📊 OPERATIONS METRICS DASHBOARD</div>',
        unsafe_allow_html=True,
    )

    if history.empty:
        st.info("No dispatches recorded yet. Run a dispatch to populate metrics.")
        return

    # Summary metrics row
    total         = len(history)
    mean_time     = history["route_time_min"].mean()
    mean_delivery = history["mesh_delivery_ratio"].mean() * 100
    critical_ct   = len(history[history["urgency_index"] >= 8.0])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Dispatches",       total)
    with c2:
        st.metric("Mean Response Time",     f"{mean_time:.1f} min")
    with c3:
        st.metric("Mesh Delivery Success",  f"{mean_delivery:.1f}%")
    with c4:
        st.metric("Critical Incidents",     critical_ct,
                  delta=f"+{critical_ct} total", delta_color="inverse")

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    # Charts row
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.plotly_chart(make_response_time_chart(history),
                        use_container_width=True, config={"displayModeBar": False})
    with col_b:
        st.plotly_chart(make_delivery_chart(history),
                        use_container_width=True, config={"displayModeBar": False})
    with col_c:
        st.plotly_chart(make_urgency_trend_chart(history),
                        use_container_width=True, config={"displayModeBar": False})

    # History table
    st.markdown('<div class="card-title" style="margin-top:0.5rem;">Dispatch History Log</div>',
                unsafe_allow_html=True)
    display_df = history[[
        "timestamp", "incident_type", "urgency_index",
        "vehicle_dispatched", "route_time_min", "mesh_delivery_ratio",
    ]].copy()
    display_df.columns = [
        "Time", "Incident Type", "Urgency", "Vehicle", "ETA (min)", "Delivery %",
    ]
    display_df["Delivery %"] = (display_df["Delivery %"] * 100).round(1).astype(str) + "%"
    display_df["Urgency"] = display_df["Urgency"].round(2)
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Urgency": st.column_config.ProgressColumn(
                "Urgency", min_value=1, max_value=10, format="%.2f"
            ),
        },
    )


# ---------------------------------------------------------------------------
# Main Streamlit application
# ---------------------------------------------------------------------------

def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_session_state()

    # ── Hero Banner ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-banner">
        <div style="display:flex;align-items:center;flex-wrap:wrap;gap:0.5rem;">
            <div>
                <div class="hero-title">🚨 GuardianNodes RoadSoS</div>
                <div class="hero-subtitle">Advanced Emergency Response Intelligence Platform · v2.0.0</div>
            </div>
            <div class="hero-badge">System Online</div>
        </div>
        <div style="display:flex;gap:1.5rem;margin-top:0.8rem;flex-wrap:wrap;">
            <span style="font-size:0.72rem;color:#4a5568;">🧠 ZSC AI Triage Engine</span>
            <span style="font-size:0.72rem;color:#4a5568;">🗺️ NetworkX Dijkstra Routing</span>
            <span style="font-size:0.72rem;color:#4a5568;">📡 BLE Mesh Protocol Simulator</span>
            <span style="font-size:0.72rem;color:#4a5568;">📊 Real-Time Operations Dashboard</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="content-wrapper">', unsafe_allow_html=True)

    # ── Load engines ─────────────────────────────────────────────────────
    with st.spinner("⚙️ Initialising AI engines and road network …"):
        triage_engine, routing_engine, mesh_sim = load_engines()
    engines = (triage_engine, routing_engine, mesh_sim)

    # ── Auto demo on first load ───────────────────────────────────────────
    if not st.session_state.demo_run:
        st.session_state.demo_run = True
        scenario = DEMO_SCENARIOS[0]
        run_full_dispatch(
            witness_text=scenario["text"],
            lat=scenario["lat"],
            lon=scenario["lon"],
            vehicle_type=scenario["vehicle"],
            engines=engines,
        )
        st.toast("✅ Auto-demo scenario executed on initialisation", icon="🚨")

    # ── Input Panel ───────────────────────────────────────────────────────
    with st.container():
        st.markdown('<div class="glass-card animate-fade-up">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📝 INCIDENT REPORT</div>', unsafe_allow_html=True)

        col_input, col_coords, col_vehicle = st.columns([3, 1.5, 1.5])

        with col_input:
            witness_text = st.text_area(
                "Witness Statement",
                value=st.session_state._prefill_text,
                height=110,
                placeholder="Describe the incident in detail…",
                label_visibility="collapsed",
            )

        with col_coords:
            lat = st.number_input(
                "Latitude", value=st.session_state._prefill_lat, format="%.6f",
                min_value=20.0, max_value=35.0, step=0.001,
            )
            lon = st.number_input(
                "Longitude", value=st.session_state._prefill_lon, format="%.6f",
                min_value=68.0, max_value=90.0, step=0.001,
            )

        with col_vehicle:
            vehicle_options = list(VEHICLE_PROFILES.keys())
            _prefill_veh = st.session_state._prefill_vehicle
            _veh_index = vehicle_options.index(_prefill_veh) if _prefill_veh in vehicle_options else 0
            vehicle_idx = st.selectbox(
                "Dispatch Vehicle",
                options=vehicle_options,
                index=_veh_index,
                format_func=lambda v: VEHICLE_PROFILES[v]["label"],
            )
            st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)

            # Demo scenario buttons
            st.caption("🎬 Quick Demo Scenarios")
            for i, sc in enumerate(DEMO_SCENARIOS):
                label_short = ["💥 Explosion", "🚗 Entrapment", "⚡ Minor Crash", "☣️ HazMat"][i]
                if st.button(label_short, key=f"demo_btn_{i}", use_container_width=True):
                    # Write to prefill keys only — never touch widget keys after render
                    st.session_state._prefill_text    = sc["text"]
                    st.session_state._prefill_lat     = sc["lat"]
                    st.session_state._prefill_lon     = sc["lon"]
                    st.session_state._prefill_vehicle = sc["vehicle"]
                    run_full_dispatch(
                        witness_text=sc["text"],
                        lat=sc["lat"],
                        lon=sc["lon"],
                        vehicle_type=sc["vehicle"],
                        engines=engines,
                    )
                    st.rerun()

        # Main dispatch button
        dispatch_col, _ = st.columns([1, 3])
        with dispatch_col:
            if st.button("🚨 DISPATCH EMERGENCY RESPONSE", key="dispatch_btn", use_container_width=True):
                if witness_text.strip():
                    run_full_dispatch(
                        witness_text=witness_text,
                        lat=lat,
                        lon=lon,
                        vehicle_type=vehicle_idx,
                        engines=engines,
                    )
                    st.rerun()
                else:
                    st.warning("⚠ Please enter a witness statement.")

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Results Panels ────────────────────────────────────────────────────
    if (
        st.session_state.last_triage is not None
        and st.session_state.last_route is not None
        and st.session_state.last_mesh is not None
    ):
        col_triage, col_route, col_mesh = st.columns([1.05, 1.2, 1.0])

        with col_triage:
            st.markdown('<div class="glass-card animate-fade-up">', unsafe_allow_html=True)
            render_triage_panel(st.session_state.last_triage)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_route:
            st.markdown('<div class="glass-card animate-fade-up">', unsafe_allow_html=True)
            render_routing_panel(st.session_state.last_route, routing_engine)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_mesh:
            st.markdown('<div class="glass-card animate-fade-up">', unsafe_allow_html=True)
            render_mesh_panel(st.session_state.last_mesh)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Metrics Dashboard ─────────────────────────────────────────────────
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    render_metrics_dashboard(st.session_state.dispatch_history)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # content-wrapper

    # Footer
    st.markdown("""
    <div style="text-align:center;padding:1rem;color:#2d3748;font-size:0.68rem;
                border-top:1px solid rgba(255,255,255,0.03);margin-top:1rem;">
        GuardianNodes RoadSoS v2.0 · Zero-Shot AI · NetworkX Routing · BLE Mesh Simulation
        · Built with Python 3.11 + Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
