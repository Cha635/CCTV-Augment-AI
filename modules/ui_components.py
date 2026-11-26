"""
UI Components and Styling for CCTV Augment AI.
Provides modern dark-themed styling, cyber-surveillance visual identity,
accessible high-contrast text typography, custom cards, and status indicators.
"""

import streamlit as st

def apply_custom_css():
    """Inject custom CSS for dark surveillance aesthetic with high-contrast, accessible typography."""
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
        
        :root {
            --bg-main: #0B1120;
            --bg-secondary: #1D2B42;
            --bg-card: #151F32;
            --border-card: #2E4166;
            --border-subtle: #1E293B;
            --text-primary: #F8FAFC;
            --text-secondary: #CBD5E1;
            --text-muted: #A3B3C7;
            --accent-cyan: #22D3EE;
            --accent-blue: #3B82F6;
            --accent-green: #22C55E;
            --accent-amber: #FBBF24;
            --accent-red: #F87171;
        }

        /* Global App Background & Base Typography */
        .stApp {
            background: linear-gradient(135deg, #070B14 0%, #0B1120 50%, #0E1726 100%) !important;
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
            color: #F8FAFC !important;
        }

        /* All Headings */
        h1, h2, h3, h4, h5, h6 {
            color: #F8FAFC !important;
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-weight: 700 !important;
            letter-spacing: -0.3px;
        }

        p, span, label, div {
            color: #CBD5E1;
        }

        strong, b {
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }

        /* Streamlit Captions & Helper text */
        .stCaption, [data-testid="stImageCaption"], [data-testid="stCaptionContainer"] {
            color: #CBD5E1 !important;
            font-size: 0.85rem !important;
            font-weight: 500 !important;
            margin-top: 4px !important;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0A101D !important;
            border-right: 1px solid #1E293B !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] {
            padding-top: 1rem;
        }
        section[data-testid="stSidebar"] .stMarkdown h1, 
        section[data-testid="stSidebar"] .stMarkdown h2, 
        section[data-testid="stSidebar"] .stMarkdown h3 {
            color: #22D3EE !important;
            font-family: 'JetBrains Mono', monospace !important;
            letter-spacing: -0.5px;
        }
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span {
            color: #CBD5E1 !important;
        }

        /* Sidebar Radio Navigation Items */
        section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
            background: #111C2E !important;
            border: 1px solid #1E293B !important;
            border-radius: 8px !important;
            padding: 7px 12px !important;
            margin-bottom: 5px !important;
            transition: all 0.2s ease !important;
            display: flex !important;
            align-items: center !important;
            cursor: pointer !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
            background: #1D2B42 !important;
            border-color: #22D3EE !important;
            transform: translateX(2px) !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] {
            background: linear-gradient(90deg, rgba(34, 211, 238, 0.2) 0%, rgba(59, 130, 246, 0.12) 100%) !important;
            border: 1px solid #22D3EE !important;
            box-shadow: 0 0 10px rgba(34, 211, 238, 0.2) !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stRadio"] label p,
        section[data-testid="stSidebar"] div[data-testid="stRadio"] label span {
            color: #CBD5E1 !important;
            font-size: 0.86rem !important;
            font-weight: 500 !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] p,
        section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] span {
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }

        /* Custom Cards */
        .cctv-card {
            background: #151F32;
            border: 1px solid #2E4166;
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.55);
            transition: all 0.2s ease-in-out;
        }
        .cctv-card:hover {
            border-color: #38BDF8;
            box-shadow: 0 12px 32px -4px rgba(34, 211, 238, 0.18);
        }
        .cctv-card h4 {
            color: #22D3EE !important;
            font-size: 1.15rem !important;
            font-weight: 700 !important;
            margin-top: 0 !important;
            margin-bottom: 0.75rem !important;
        }
        .cctv-card p, .cctv-card li {
            color: #CBD5E1 !important;
            line-height: 1.6;
        }

        /* Metric Card with strictly uniform height and alignment */
        .metric-card {
            background: linear-gradient(145deg, #151F32 0%, #101929 100%);
            border: 1px solid #2E4166;
            border-radius: 12px;
            padding: 1.1rem 1.25rem;
            position: relative;
            overflow: hidden;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
            transition: transform 0.2s ease, border-color 0.2s ease;
            min-height: 135px;
            height: 135px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: #38BDF8;
        }
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: linear-gradient(180deg, #22D3EE 0%, #3B82F6 100%);
        }
        .metric-value {
            font-size: 2.1rem;
            font-weight: 800;
            color: #FFFFFF !important;
            font-family: 'JetBrains Mono', monospace;
            line-height: 1.1;
            margin: 0.15rem 0;
        }
        .metric-label {
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: #A3B3C7 !important;
            font-weight: 700;
            margin: 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .metric-sub {
            font-size: 0.78rem;
            color: #22D3EE !important;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 500;
            margin: 0;
            line-height: 1.3;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }

        /* Banner Header */
        .header-banner {
            background: linear-gradient(90deg, rgba(34, 211, 238, 0.15) 0%, rgba(59, 130, 246, 0.1) 50%, rgba(15, 23, 42, 0.4) 100%);
            border-left: 5px solid #22D3EE;
            border-top: 1px solid rgba(34, 211, 238, 0.2);
            border-bottom: 1px solid rgba(34, 211, 238, 0.2);
            border-radius: 0 12px 12px 0;
            padding: 1.3rem 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        }
        .header-title {
            font-size: 1.95rem;
            font-weight: 800;
            color: #FFFFFF !important;
            margin: 0;
            letter-spacing: -0.5px;
        }
        .header-subtitle {
            font-size: 0.98rem;
            color: #CBD5E1 !important;
            margin: 0.45rem 0 0 0;
            font-weight: 400;
        }

        /* Status Badge */
        .badge-live {
            display: inline-flex;
            align-items: center;
            background: rgba(34, 197, 94, 0.18);
            border: 1px solid rgba(34, 197, 94, 0.5);
            color: #4ADE80 !important;
            padding: 0.3rem 0.85rem;
            border-radius: 9999px;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.8px;
            text-transform: uppercase;
            font-family: 'JetBrains Mono', monospace;
            box-shadow: 0 0 10px rgba(34, 197, 94, 0.2);
        }
        .badge-live::before {
            content: '';
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #22C55E;
            margin-right: 7px;
            box-shadow: 0 0 8px #22C55E;
        }

        .badge-cctv {
            display: inline-flex;
            align-items: center;
            background: rgba(34, 211, 238, 0.18);
            border: 1px solid rgba(34, 211, 238, 0.5);
            color: #22D3EE !important;
            padding: 0.3rem 0.85rem;
            border-radius: 9999px;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.8px;
            text-transform: uppercase;
            font-family: 'JetBrains Mono', monospace;
            box-shadow: 0 0 10px rgba(34, 211, 238, 0.2);
        }

        /* Buttons Styling */
        .stButton>button {
            background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
            color: #FFFFFF !important;
            border: 1px solid #38BDF8 !important;
            border-radius: 8px !important;
            padding: 0.6rem 1.3rem !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35) !important;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #22D3EE 0%, #0284C7 100%) !important;
            border-color: #67E8F9 !important;
            color: #070B14 !important;
            box-shadow: 0 6px 22px rgba(34, 211, 238, 0.5) !important;
            transform: translateY(-1px) !important;
        }
        .stButton>button:active {
            transform: translateY(0px) !important;
        }

        /* Download Buttons Standout */
        .stDownloadButton>button {
            background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
            color: #FFFFFF !important;
            border: 1px solid #34D399 !important;
            border-radius: 8px !important;
            padding: 0.6rem 1.3rem !important;
            font-weight: 700 !important;
            font-size: 0.9rem !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 4px 14px rgba(5, 150, 105, 0.35) !important;
        }
        .stDownloadButton>button:hover {
            background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
            border-color: #6EE7B7 !important;
            box-shadow: 0 6px 22px rgba(16, 185, 129, 0.5) !important;
            transform: translateY(-1px) !important;
        }

        /* Inputs, Selectboxes, Number Inputs & Dropdowns */
        div[data-baseweb="input"] {
            background-color: #151F32 !important;
            border: 1px solid #2E4166 !important;
            border-radius: 8px !important;
        }
        div[data-baseweb="input"] input {
            color: #F8FAFC !important;
            background-color: transparent !important;
            font-size: 0.92rem !important;
        }
        div[data-baseweb="input"] input::placeholder {
            color: #A3B3C7 !important;
        }
        div[data-baseweb="input"]:focus-within {
            border-color: #22D3EE !important;
            box-shadow: 0 0 0 2px rgba(34, 211, 238, 0.25) !important;
        }

        /* Selectbox trigger & Popovers */
        div[data-baseweb="select"] {
            background-color: #151F32 !important;
            border: 1px solid #2E4166 !important;
            border-radius: 8px !important;
        }
        div[data-baseweb="select"] * {
            color: #F8FAFC !important;
            background-color: transparent !important;
        }
        div[data-baseweb="popover"], div[data-baseweb="popover"] ul {
            background-color: #151F32 !important;
            border: 1px solid #2E4166 !important;
            border-radius: 8px !important;
        }
        li[role="option"] {
            color: #F8FAFC !important;
            background-color: #151F32 !important;
            font-size: 0.9rem !important;
            padding: 8px 14px !important;
        }
        li[role="option"]:hover, li[aria-selected="true"] {
            background-color: #1D2B42 !important;
            color: #22D3EE !important;
            font-weight: 600 !important;
        }

        /* Checkbox */
        div[data-testid="stCheckbox"] label {
            cursor: pointer;
        }
        div[data-testid="stCheckbox"] label span {
            color: #CBD5E1 !important;
            font-weight: 500 !important;
            font-size: 0.9rem !important;
        }

        /* Sliders */
        div[data-testid="stSlider"] label {
            color: #CBD5E1 !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
        }
        div[data-testid="stSlider"] div[data-testid="stMarkdownContainer"] p {
            color: #CBD5E1 !important;
        }
        div[data-testid="stSlider"] [role="slider"] {
            background-color: #22D3EE !important;
            box-shadow: 0 0 8px #22D3EE !important;
        }

        /* File Uploader */
        div[data-testid="stFileUploader"] section {
            background-color: #151F32 !important;
            border: 2px dashed #2E4166 !important;
            border-radius: 12px !important;
            padding: 1.5rem !important;
        }
        div[data-testid="stFileUploader"] section:hover {
            border-color: #22D3EE !important;
        }
        div[data-testid="stFileUploader"] section span,
        div[data-testid="stFileUploader"] section p,
        div[data-testid="stFileUploader"] section small {
            color: #CBD5E1 !important;
        }
        div[data-testid="stFileUploader"] section button {
            background: #1D2B42 !important;
            border: 1px solid #2E4166 !important;
            color: #F8FAFC !important;
        }

        /* Code block and logs */
        .log-box {
            background: #070B14;
            border: 1px solid #2E4166;
            border-radius: 8px;
            padding: 1.1rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.84rem;
            color: #38BDF8;
            max-height: 400px;
            overflow-y: auto;
            white-space: pre-wrap;
            line-height: 1.6;
        }

        /* Tabs customization */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: transparent;
            border-bottom: 1px solid #2E4166 !important;
        }
        .stTabs [data-baseweb="tab"] {
            background: #151F32;
            border: 1px solid #2E4166;
            border-bottom: none;
            border-radius: 8px 8px 0 0;
            color: #CBD5E1 !important;
            padding: 0.6rem 1.4rem;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background: #1D2B42 !important;
            color: #FFFFFF !important;
        }
        .stTabs [aria-selected="true"] {
            background: #1D2B42 !important;
            border-bottom: 3px solid #22D3EE !important;
            color: #22D3EE !important;
            font-weight: 700 !important;
        }

        /* Expanders */
        div[data-testid="stExpander"] {
            background: #151F32 !important;
            border: 1px solid #2E4166 !important;
            border-radius: 10px !important;
            margin-bottom: 1rem !important;
        }
        div[data-testid="stExpander"] summary {
            color: #F8FAFC !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
        }
        div[data-testid="stExpander"] summary:hover {
            color: #22D3EE !important;
        }
        div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
            background: #0E1726 !important;
            padding: 1rem !important;
            border-top: 1px solid #1E293B !important;
        }

        /* Native Alert Callouts */
        div[data-testid="stAlert"] {
            border-radius: 8px !important;
            font-size: 0.9rem !important;
            font-weight: 500 !important;
        }
        div[data-testid="stAlert"] p {
            color: #F8FAFC !important;
        }

        /* Tables */
        table {
            width: 100%;
            border-collapse: collapse;
            background-color: #151F32 !important;
            border-radius: 8px;
            overflow: hidden;
            margin: 1rem 0;
            border: 1px solid #2E4166;
        }
        th {
            background-color: #1D2B42 !important;
            color: #22D3EE !important;
            font-weight: 700 !important;
            padding: 10px 14px !important;
            text-align: left;
            border-bottom: 2px solid #2E4166 !important;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
        }
        td {
            padding: 9px 14px !important;
            color: #CBD5E1 !important;
            border-bottom: 1px solid #1E293B !important;
            font-size: 0.88rem;
        }
        tr:hover td {
            background-color: rgba(34, 211, 238, 0.05) !important;
        }

        /* Streamlit DataFrames & Data Tables */
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            background-color: #151F32 !important;
            border: 1px solid #2E4166 !important;
            border-radius: 8px !important;
            color: #F8FAFC !important;
        }
        [data-testid="stDataFrame"] div, [data-testid="stDataFrame"] span {
            color: #F8FAFC !important;
        }
        
        /* Secondary Buttons & Custom Button Types */
        button[kind="secondary"] {
            background: #1D2B42 !important;
            color: #F8FAFC !important;
            border: 1px solid #2E4166 !important;
        }
        button[kind="secondary"]:hover {
            background: #2E4166 !important;
            border-color: #22D3EE !important;
            color: #22D3EE !important;
        }

        /* Plotly Containers */
        [data-testid="stPlotlyChart"] {
            border-radius: 10px !important;
            overflow: hidden !important;
        }

        /* Inline code tags */
        code {
            background: #1D2B42 !important;
            color: #22D3EE !important;
            padding: 2px 6px !important;
            border-radius: 4px !important;
            border: 1px solid #2E4166 !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.85em !important;
        }
        </style>
    """, unsafe_allow_html=True)

def render_header(title: str, subtitle: str, badge: str = "SYSTEM ONLINE"):
    """Render consistent styled header banner across all pages."""
    badge_html = f'<span class="badge-live">{badge}</span>' if badge else ''
    st.markdown(f"""
        <div class="header-banner">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <h1 class="header-title">{title}</h1>
                {badge_html}
            </div>
            <p class="header-subtitle">{subtitle}</p>
        </div>
    """, unsafe_allow_html=True)

def render_metric_card(label: str, value: str, subtext: str = "", border_color: str = "#22D3EE"):
    """Render a surveillance-styled KPI metric card with accessible high-contrast text."""
    st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid {border_color};">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtext}</div>
        </div>
    """, unsafe_allow_html=True)

def render_alert_box(text: str, level: str = "info"):
    """Render custom alert box with accessible high contrast."""
    colors = {
        "info": ("rgba(34, 211, 238, 0.15)", "#22D3EE", "#E0F2FE"),
        "success": ("rgba(34, 197, 94, 0.15)", "#22C55E", "#DCFCE7"),
        "warning": ("rgba(251, 191, 36, 0.15)", "#FBBF24", "#FEF3C7"),
        "error": ("rgba(248, 113, 113, 0.15)", "#F87171", "#FEE2E2"),
    }
    bg, border, text_color = colors.get(level, colors["info"])
    st.markdown(f"""
        <div style="background: {bg}; border: 1px solid {border}; border-radius: 8px; padding: 0.95rem 1.3rem; color: {text_color}; margin: 0.9rem 0; font-size: 0.9rem; font-weight: 500; line-height: 1.5;">
            {text}
        </div>
    """, unsafe_allow_html=True)
