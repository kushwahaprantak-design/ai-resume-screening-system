import streamlit as st

def inject_custom_css():
    """
    Inject professional, human-designed SaaS dark UI system.
    Inspired by Linear, Vercel, and Supabase dashboards.
    - True zinc-black neutrals (no purple/navy gradients)
    - Single indigo accent, used sparingly
    - Solid borders (no glow, no blur)
    - Clean typography, proper whitespace
    """
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

        /* ─── GLOBAL RESET ─────────────────────────────── */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        /* ─── MAIN CANVAS ───────────────────────────────── */
        .stApp {
            background-color: #09090b !important;
            color: #fafafa !important;
        }

        /* ─── SIDEBAR ───────────────────────────────────── */
        [data-testid="stSidebar"] {
            background-color: #111111 !important;
            border-right: 1px solid #27272a !important;
        }

        [data-testid="stSidebar"] .stMarkdown h3,
        [data-testid="stSidebar"] .stMarkdown h2 {
            font-size: 0.7rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.1em !important;
            color: #71717a !important;
            margin-bottom: 0.5rem !important;
        }

        [data-testid="stSidebar"] hr {
            border-color: #27272a !important;
            margin: 1rem 0 !important;
        }

        /* Sidebar radio nav items */
        div[data-testid="stSidebar"] div[role="radiogroup"] label {
            border-radius: 6px;
            padding: 0.45rem 0.65rem !important;
            margin-bottom: 2px;
            transition: background 0.12s ease;
            color: #a1a1aa !important;
            font-size: 0.9rem !important;
            font-weight: 400 !important;
        }

        div[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: #1c1c1f !important;
            color: #fafafa !important;
        }

        div[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"],
        div[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: #1c1c1f !important;
            color: #fafafa !important;
        }

        /* Sidebar slider labels & text */
        [data-testid="stSidebar"] .stSlider p,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label {
            color: #a1a1aa !important;
            font-size: 0.875rem !important;
        }

        /* ─── HERO HEADER ───────────────────────────────── */
        .enterprise-header {
            background-color: #111111;
            border: 1px solid #27272a;
            border-radius: 8px;
            padding: 1.75rem 2rem;
            margin-bottom: 1.75rem;
        }

        .header-title {
            color: #fafafa !important;
            -webkit-text-fill-color: #fafafa !important;
            background: none !important;
            font-size: 1.75rem;
            font-weight: 700;
            margin: 0 0 0.4rem 0;
            letter-spacing: -0.025em;
            line-height: 1.3;
        }

        .header-subtitle {
            color: #71717a;
            font-size: 0.9rem;
            font-weight: 400;
            margin: 0;
            line-height: 1.6;
        }

        /* ─── TECH BADGE PILLS ──────────────────────────── */
        .hero-tech-badge {
            display: inline-flex;
            align-items: center;
            background-color: #1c1c1f;
            color: #a1a1aa;
            border: 1px solid #27272a;
            border-radius: 5px;
            padding: 0.2rem 0.65rem;
            font-size: 0.73rem;
            font-weight: 500;
            margin-right: 0.4rem;
            margin-top: 0.6rem;
            letter-spacing: 0.01em;
        }

        /* ─── CARDS & CONTAINERS ────────────────────────── */
        .card-box, .section-card {
            background-color: #111111;
            border: 1px solid #27272a;
            border-radius: 8px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.25rem;
        }

        /* ─── KPI METRIC CARDS ──────────────────────────── */
        .metric-card {
            background-color: #111111;
            border: 1px solid #27272a;
            border-radius: 8px;
            padding: 1.1rem 1.25rem;
            text-align: center;
            transition: border-color 0.15s ease;
        }

        .metric-card:hover {
            border-color: #3f3f46;
        }

        .metric-value {
            font-size: 1.875rem;
            font-weight: 700;
            color: #fafafa;
            margin-top: 0.2rem;
            letter-spacing: -0.02em;
            -webkit-text-fill-color: #fafafa !important;
            background: none !important;
        }

        .metric-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: #71717a;
            font-weight: 500;
        }

        /* ─── SKILL PILL BADGES ─────────────────────────── */
        .skill-pill {
            display: inline-block;
            background-color: #16a34a1a;
            color: #4ade80;
            border: 1px solid #166534;
            border-radius: 4px;
            padding: 0.22rem 0.65rem;
            margin: 0.18rem;
            font-size: 0.8rem;
            font-weight: 500;
        }

        .skill-pill-missing {
            background-color: #dc26261a;
            color: #f87171;
            border: 1px solid #7f1d1d;
            border-radius: 4px;
            padding: 0.22rem 0.65rem;
            margin: 0.18rem;
            font-size: 0.8rem;
            font-weight: 500;
        }

        /* ─── BUTTONS ───────────────────────────────────── */
        .stButton>button {
            background-color: #6366f1 !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 6px !important;
            padding: 0.5rem 1.1rem !important;
            font-size: 0.875rem !important;
            font-weight: 500 !important;
            letter-spacing: 0.01em !important;
            box-shadow: none !important;
            transition: background-color 0.12s ease, opacity 0.12s ease !important;
        }

        .stButton>button:hover {
            background-color: #4f46e5 !important;
            transform: none !important;
            box-shadow: none !important;
        }

        .stButton>button:active {
            background-color: #4338ca !important;
            opacity: 0.95 !important;
        }

        /* Secondary/outline-ish buttons (download etc.) */
        .stDownloadButton>button {
            background-color: #1c1c1f !important;
            color: #fafafa !important;
            border: 1px solid #27272a !important;
            border-radius: 6px !important;
            padding: 0.5rem 1.1rem !important;
            font-size: 0.875rem !important;
            font-weight: 500 !important;
            box-shadow: none !important;
            transition: background-color 0.12s ease !important;
        }

        .stDownloadButton>button:hover {
            background-color: #27272a !important;
            border-color: #3f3f46 !important;
        }

        /* ─── INPUTS & FORM ELEMENTS ────────────────────── */
        div[data-baseweb="input"] input,
        div[data-baseweb="textarea"] textarea {
            background-color: #111111 !important;
            color: #fafafa !important;
            border-radius: 6px !important;
            border: 1px solid #27272a !important;
            font-size: 0.875rem !important;
        }

        div[data-baseweb="input"] input:focus,
        div[data-baseweb="textarea"] textarea:focus {
            border-color: #6366f1 !important;
            box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
            outline: none !important;
        }

        /* Select boxes */
        div[data-baseweb="select"] > div {
            background-color: #111111 !important;
            border: 1px solid #27272a !important;
            border-radius: 6px !important;
            color: #fafafa !important;
        }

        /* ─── FILE UPLOADER ─────────────────────────────── */
        section[data-testid="stFileUploader"] {
            background-color: #111111;
            border: 1px dashed #27272a;
            border-radius: 8px;
            padding: 1.1rem;
            transition: border-color 0.15s ease;
        }

        section[data-testid="stFileUploader"]:hover {
            border-color: #6366f1;
        }

        /* ─── NAVIGATION TABS ───────────────────────────── */
        div[data-baseweb="tab-list"] {
            background-color: transparent !important;
            border-bottom: 1px solid #27272a !important;
            gap: 0 !important;
        }

        button[data-baseweb="tab"] {
            font-size: 0.875rem !important;
            font-weight: 400 !important;
            color: #71717a !important;
            border-bottom: 2px solid transparent !important;
            padding: 0.65rem 1rem !important;
            background: transparent !important;
            border-radius: 0 !important;
            transition: color 0.12s ease !important;
        }

        button[data-baseweb="tab"]:hover {
            color: #a1a1aa !important;
            background: transparent !important;
        }

        button[aria-selected="true"] {
            color: #fafafa !important;
            border-bottom: 2px solid #6366f1 !important;
            font-weight: 500 !important;
            background: transparent !important;
        }

        /* ─── EXPANDERS ─────────────────────────────────── */
        div[data-testid="stExpander"] {
            background-color: #111111 !important;
            border: 1px solid #27272a !important;
            border-radius: 8px !important;
            overflow: hidden !important;
        }

        div[data-testid="stExpander"] summary {
            color: #a1a1aa !important;
            font-size: 0.875rem !important;
            font-weight: 500 !important;
        }

        /* ─── ALERTS / INFO BOXES ───────────────────────── */
        div[data-testid="stAlert"] {
            border-radius: 6px !important;
            border: 1px solid #27272a !important;
            background-color: #111111 !important;
        }

        /* ─── CHAT MESSAGES ─────────────────────────────── */
        div[data-testid="stChatMessage"] {
            background-color: #111111 !important;
            border: 1px solid #27272a !important;
            border-radius: 8px !important;
        }

        /* ─── DATAFRAMES / TABLES ───────────────────────── */
        .stDataFrame {
            border: 1px solid #27272a !important;
            border-radius: 8px !important;
            overflow: hidden !important;
        }

        /* ─── SCROLLBAR ─────────────────────────────────── */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }

        ::-webkit-scrollbar-track {
            background: #09090b;
        }

        ::-webkit-scrollbar-thumb {
            background: #27272a;
            border-radius: 3px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #3f3f46;
        }

        /* ─── HIDE STREAMLIT CHROME ─────────────────────── */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header[data-testid="stHeader"] {
            background-color: #09090b !important;
            border-bottom: 1px solid #27272a !important;
        }

        /* ─── OVERFLOW / RESPONSIVE ─────────────────────── */
        .stDataFrame, .stPlotlyChart, .stTable, pre, code {
            overflow-x: auto !important;
            max-width: 100% !important;
        }

        @media (max-width: 768px) {
            .enterprise-header {
                padding: 1.1rem 1rem;
            }
            .header-title {
                font-size: 1.35rem;
            }
            .card-box, .section-card {
                padding: 1rem;
            }
            div[data-testid="stHorizontalBlock"] {
                flex-direction: column !important;
            }
            div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
                width: 100% !important;
                flex: 1 1 100% !important;
            }
        }
        </style>
    """, unsafe_allow_html=True)


def apply_custom_css():
    """Alias for inject_custom_css for backward compatibility."""
    inject_custom_css()
