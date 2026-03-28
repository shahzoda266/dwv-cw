import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import math
import matplotlib.pyplot as plt
import io
import json
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(
    page_title="Data Wrangler",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= GLOBAL STYLES =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #0f0f17 !important;
    color: #e2e2f0 !important;
}

.main .block-container {
    background: #0f0f17 !important;
    padding-top: 2rem !important;
}

section[data-testid="stSidebar"] {
    background: #13131f !important;
    border-right: 1px solid #2a2a3d !important;
}
section[data-testid="stSidebar"] * { color: #e2e2f0 !important; }

/* Expanders */
details {
    background: #16161f !important;
    border: 1px solid #2a2a3d !important;
    border-radius: 14px !important;
    margin-bottom: 12px !important;
    overflow: hidden !important;
}
details:hover { border-color: #4f46e5 !important; }
details[open] {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 1px #4f46e520, 0 8px 32px #4f46e510 !important;
}
details summary {
    font-family: 'Space Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    color: #c8c8e8 !important;
    padding: 16px 20px !important;
    cursor: pointer !important;
    letter-spacing: 0.5px !important;
    list-style: none !important;
}
details summary:hover { color: #a78bfa !important; }
details summary::-webkit-details-marker { display: none !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 12px #4f46e530 !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px #4f46e550 !important;
}

/* Inputs */
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    background: #1e1e2e !important;
    border: 1px solid #2a2a3d !important;
    border-radius: 8px !important;
    color: #e2e2f0 !important;
}
.stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 2px #4f46e520 !important;
}

/* Selectbox & Multiselect */
.stSelectbox > div > div, .stMultiSelect > div > div {
    background: #1e1e2e !important;
    border: 1px solid #2a2a3d !important;
    border-radius: 8px !important;
    color: #e2e2f0 !important;
}

/* Slider */
.stSlider > div > div > div > div { background: #4f46e5 !important; }

/* Dataframe */
.stDataFrame { border: 1px solid #2a2a3d !important; border-radius: 10px !important; overflow: hidden !important; }
thead tr th {
    background-color: #1e1e2e !important;
    color: #a78bfa !important;
    font-weight: 700 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: #16161f !important;
    border: 1px solid #2a2a3d !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
}
[data-testid="metric-container"] label { color: #6060a0 !important; font-size: 11px !important; text-transform: uppercase !important; letter-spacing: 1.5px !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #a78bfa !important; font-family: 'Space Mono', monospace !important; font-size: 28px !important; }

/* Alerts */
div[data-testid="stAlert"] { border-radius: 8px !important; }

/* Radio & Checkbox labels */
.stRadio label, .stCheckbox label { color: #c8c8e8 !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: #16161f !important;
    border: 2px dashed #2a2a3d !important;
    border-radius: 14px !important;
    padding: 10px !important;
}
[data-testid="stFileUploader"]:hover { border-color: #4f46e5 !important; }

/* Download button */
.stDownloadButton > button {
    background: #16161f !important;
    color: #a78bfa !important;
    border: 1px solid #4f46e5 !important;
    border-radius: 8px !important;
}

/* Divider */
hr { border-color: #2a2a3d !important; }

/* General text */
p, label, span, div { color: #c8c8e8; }
h1, h2, h3, h4 { font-family: 'Space Mono', monospace !important; color: #e2e2f0 !important; }
</style>
""", unsafe_allow_html=True)

# ================= INIT STATE =================
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "just_reset" not in st.session_state:
    st.session_state.just_reset = False
if "log" not in st.session_state:
    st.session_state.log = []

if "df" in st.session_state:
    df = st.session_state.df

# ================= SIDEBAR =================
st.sidebar.markdown("""
<div style='padding:24px 0 8px 0'>
    <p style='font-family:Space Mono,monospace;font-size:11px;color:#4f46e5;letter-spacing:3px;text-transform:uppercase;margin:0 0 16px 0'>Navigate</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.selectbox(
    "",
    ["Upload & Overview", "Cleaning & Preparation", "Visualization Builder", "Export & Report"],
    label_visibility="collapsed"
)

st.sidebar.markdown("<hr style='border-color:#2a2a3d;margin:16px 0'>", unsafe_allow_html=True)

if "df" in st.session_state:
    _df = st.session_state.df
    st.sidebar.markdown(f"""
    <p style='font-family:Space Mono,monospace;font-size:10px;color:#4f46e5;letter-spacing:2px;text-transform:uppercase;margin:0 0 12px 0'>Dataset</p>
    <div style='display:flex;flex-direction:column;gap:8px'>
        <div style='background:#1e1e2e;border-radius:8px;padding:10px 14px;border-left:3px solid #4f46e5'>
            <span style='color:#6060a0;font-size:10px;text-transform:uppercase;letter-spacing:1px'>Rows</span><br>
            <span style='color:#a78bfa;font-family:Space Mono,monospace;font-size:18px;font-weight:700'>{_df.shape[0]:,}</span>
        </div>
        <div style='background:#1e1e2e;border-radius:8px;padding:10px 14px;border-left:3px solid #7c3aed'>
            <span style='color:#6060a0;font-size:10px;text-transform:uppercase;letter-spacing:1px'>Columns</span><br>
            <span style='color:#a78bfa;font-family:Space Mono,monospace;font-size:18px;font-weight:700'>{_df.shape[1]}</span>
        </div>
        <div style='background:#1e1e2e;border-radius:8px;padding:10px 14px;border-left:3px solid #ef4444'>
            <span style='color:#6060a0;font-size:10px;text-transform:uppercase;letter-spacing:1px'>Missing</span><br>
            <span style='color:#f87171;font-family:Space Mono,monospace;font-size:18px;font-weight:700'>{_df.isnull().sum().sum():,}</span>
        </div>
        <div style='background:#1e1e2e;border-radius:8px;padding:10px 14px;border-left:3px solid #f59e0b'>
            <span style='color:#6060a0;font-size:10px;text-transform:uppercase;letter-spacing:1px'>Duplicates</span><br>
            <span style='color:#fcd34d;font-family:Space Mono,monospace;font-size:18px;font-weight:700'>{_df.duplicated().sum():,}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<div style='margin-bottom:32px'>
    <h1 style='font-family:Space Mono,monospace;font-size:34px;font-weight:700;color:#e2e2f0;margin:0;line-height:1.1'>
        Data Wrangler <span style='color:#a78bfa'>&</span> Visualizer
    </h1>
    <p style='color:#7a7ac9;font-size:17px;margin:8px 0 0 0'>Clean, transform, and visualize your data effortlessly</p>
</div>
""", unsafe_allow_html=True)



# ================= PAGE A =================
if page == "Upload & Overview":

    @st.cache_data
    def load_data(uploaded_file):
        file_name = uploaded_file.name.lower()

        try:
            uploaded_file.seek(0)

            if file_name.endswith(".csv"):
                return pd.read_csv(uploaded_file)

            elif file_name.endswith(".xlsx"):
                return pd.read_excel(uploaded_file)

            elif file_name.endswith(".json"):
                return pd.read_json(uploaded_file)

            else:
                raise ValueError("Unsupported file format. Please upload CSV, Excel, or JSON.")

        except Exception as e:
            raise ValueError(f"Could not read the uploaded file: {e}")

    @st.cache_data
    def profile_data(df):
        def simplify_dtype(dtype):
            if "int" in str(dtype) or "float" in str(dtype):
                return "Numeric"
            elif "datetime" in str(dtype):
                return "Date"
            else:
                return "Text"

        col_types = pd.DataFrame({
            "Column": df.columns,
            "Type": [simplify_dtype(dtype) for dtype in df.dtypes]
        })

        num_df = df.select_dtypes(include=np.number)
        cat_df = df.select_dtypes(include=["object", "category"])

        numeric_summary = num_df.describe() if not num_df.empty else pd.DataFrame()

        categorical_summary = pd.DataFrame({
            "Column": cat_df.columns,
            "Unique Values": [cat_df[col].nunique() for col in cat_df.columns],
            "Most Frequent": [
                cat_df[col].mode()[0] if not cat_df[col].mode().empty else None
                for col in cat_df.columns
            ]
        }) if not cat_df.empty else pd.DataFrame()

        missing = df.isnull().sum()
        percent = (missing / len(df)) * 100 if len(df) > 0 else 0

        return {
            "col_types": col_types,
            "numeric_summary": numeric_summary,
            "categorical_summary": categorical_summary,
            "missing_summary": pd.DataFrame({"Missing": missing, "%": percent}),
            "duplicate_count": int(df.duplicated().sum())
        }

    cols_feat = st.columns(6)
    features = [
        ("📂", "Upload", "CSV, Excel, JSON"),
        ("🧹", "Clean", "Missing & duplicates"),
        ("📊", "Visualize", "Interactive charts"),
        ("🔄", "Transform", "Scale & modify"),
        ("⚙️", "Process", "Advanced ops"),
        ("📤", "Export", "Download results"),
    ]

    for i, (icon, title, sub) in enumerate(features):
        with cols_feat[i]:
            st.markdown(f"""
            <div style='background:#16161f;border:1px solid #2a2a3d;border-radius:12px;padding:16px 10px;text-align:center;margin-bottom:24px'>
                <div style='font-size:22px;margin-bottom:6px'>{icon}</div>
                <div style='font-family:Space Mono,monospace;font-size:12px;font-weight:700;color:#c8c8e8'>{title}</div>
                <div style='font-size:12px;color:#8a8ad0;margin-top:4px'>{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <p style='font-family:Space Mono,monospace;font-size:11px;color:#4f46e5;letter-spacing:2px;text-transform:uppercase;margin:0 0 8px 0'>Step 01</p>
    <h2 style='font-family:Space Mono,monospace;font-size:20px;color:#e2e2f0;margin:0 0 16px 0'>Upload Your Dataset</h2>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop your file here — CSV, Excel, or JSON",
        type=["csv", "xlsx", "json"],
        key=f"file_uploader_{st.session_state.uploader_key}"
    )

    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            st.session_state.df = df.copy()
            st.session_state.original_df = df.copy()
            st.success("Dataset loaded successfully.")
        except Exception as e:
            st.error(str(e))
    st.markdown("<hr style='border-color:#dddddd;margin:24px 0'>", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-family:Space Mono,monospace;font-size:11px;color:#9b1c2e;letter-spacing:2px;text-transform:uppercase;margin:0 0 8px 0'>Optional</p>
    <h2 style='font-family:Space Mono,monospace;font-size:20px;color:#111111;margin:0 0 16px 0'>Connect Google Sheet</h2>
    """, unsafe_allow_html=True)

    sheet_url = st.text_input(
        "Paste your Google Sheet URL",
        placeholder="https://docs.google.com/spreadsheets/d/...",
        key="gsheet_url"
    )

    if st.button("Load Google Sheet", key="load_gsheet_btn"):
        if not sheet_url.strip():
            st.warning("Please enter a Google Sheet URL.")
        else:
            try:
                scopes = [
                    "https://www.googleapis.com/auth/spreadsheets.readonly",
                    "https://www.googleapis.com/auth/drive.readonly"
                ]
                creds = Credentials.from_service_account_file("credentials.json", scopes=scopes)
                client = gspread.authorize(creds)
                sheet = client.open_by_url(sheet_url)
                worksheet = sheet.get_worksheet(0)
                data = worksheet.get_all_records()
                df = pd.DataFrame(data)

                if df.empty:
                    st.warning("The sheet appears to be empty.")
                else:
                    st.session_state.df = df.copy()
                    st.session_state.original_df = df.copy()
                    st.success(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns from Google Sheets.")

            except FileNotFoundError:
                st.error("credentials.json not found. Add your Google service account key file to the project folder.")
            except gspread.exceptions.NoValidUrlKeyFound:
                st.error("Invalid Google Sheet URL. Make sure you copied the full URL.")
            except gspread.exceptions.APIError as e:
                st.error(f"Google API error: {e}")
            except Exception as e:
                st.error(f"Failed to load sheet: {e}")
    if "df" not in st.session_state:
        if st.session_state.get("just_reset", False):
            st.success("Session reset. Please upload your dataset again.")
            st.session_state.just_reset = False
    else:
        df = st.session_state.df
        profile = profile_data(df)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <p style='font-family:Space Mono,monospace;font-size:11px;color:#4f46e5;letter-spacing:2px;text-transform:uppercase;margin:0 0 8px 0'>Step 02</p>
        <h2 style='font-family:Space Mono,monospace;font-size:20px;color:#e2e2f0;margin:0 0 16px 0'>Dataset Overview</h2>
        """, unsafe_allow_html=True)

        st.caption("Preview of your uploaded data (first 50 rows)")
        st.dataframe(df.head(50), use_container_width=True, height=300)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Rows", df.shape[0])
        with c2:
            st.metric("Columns", df.shape[1])
        with c3:
            st.metric("Missing Values", int(df.isnull().sum().sum()))

        st.dataframe(profile["col_types"], use_container_width=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <p style='font-family:Space Mono,monospace;font-size:11px;color:#4f46e5;letter-spacing:2px;text-transform:uppercase;margin:16px 0 8px 0'>Step 03</p>
        <h2 style='font-family:Space Mono,monospace;font-size:20px;color:#e2e2f0;margin:0 0 16px 0'>Summary Statistics</h2>
        """, unsafe_allow_html=True)

        with st.expander("Numeric Summary", expanded=False):
            if not profile["numeric_summary"].empty:
                st.dataframe(profile["numeric_summary"], use_container_width=True)
            else:
                st.info("No numeric columns available")

        with st.expander("Categorical Summary", expanded=False):
            if not profile["categorical_summary"].empty:
                st.dataframe(profile["categorical_summary"], use_container_width=True)
            else:
                st.info("No categorical columns available")

        with st.expander("Missing Values", expanded=False):
            st.dataframe(profile["missing_summary"], use_container_width=True)

        with st.expander("Duplicates", expanded=False):
            st.markdown(f"""
            <div style='background:#1e1e2e;border-radius:10px;padding:20px;text-align:center'>
                <p style='color:#6060a0;font-size:11px;text-transform:uppercase;letter-spacing:1.5px;margin:0 0 4px 0'>Duplicate Rows Found</p>
                <p style='color:#fcd34d;font-family:Space Mono,monospace;font-size:36px;font-weight:700;margin:0'>{profile["duplicate_count"]:,}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        if st.button("Reset Session"):
            current_key = st.session_state.get("uploader_key", 0)

            for key in list(st.session_state.keys()):
                del st.session_state[key]

            st.session_state.uploader_key = current_key + 1
            st.session_state.just_reset = True
            st.session_state.log = []
            st.rerun()



# ================= PAGE B =================
elif page == "Cleaning & Preparation":

    if "df" not in st.session_state:
        st.warning("Please upload a dataset first on the Upload & Overview page.")
    else:
        df = st.session_state.df.copy()

        if "log" not in st.session_state:
            st.session_state.log = []

        def add_log(operation, parameters, affected_columns):
            st.session_state.log.append({
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "operation": operation,
                "parameters": parameters,
                "affected_columns": affected_columns if affected_columns else []
            })

        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

        st.markdown("""
        <p style='font-family:Space Mono,monospace;font-size:11px;color:#4f46e5;letter-spacing:2px;text-transform:uppercase;margin:0 0 8px 0'>Page B</p>
        <h2 style='font-family:Space Mono,monospace;font-size:22px;color:#e2e2f0;margin:0 0 4px 0'>Cleaning Tools</h2>
        <p style='color:#6060a0;font-size:13px;margin:0 0 24px 0'>Click any section below to expand it and apply transformations.</p>
        """, unsafe_allow_html=True)

        # ===== TYPE CONVERSION =====
        with st.expander("Data Type Conversion", expanded=False):
            col = st.selectbox("Column", df.columns, key="dtype_col")
            dtype = st.selectbox("Convert to", ["Numeric", "Datetime", "Categorical"], key="dtype_target")
            fmt = st.text_input("Datetime format (optional)", key="dtype_fmt") if dtype == "Datetime" else None

            clean_option = []
            if dtype == "Numeric":
                clean_option = st.multiselect(
                    "Clean numeric issues",
                    ["Remove commas (,)", "Remove currency symbols ($, €, £)", "Remove spaces"],
                    key="dtype_clean_option"
                )

            if st.button("Apply Conversion", key="apply_conversion_btn"):
                try:
                    before_dtype = str(df[col].dtype)

                    if dtype == "Numeric":
                        temp = df[col].astype(str)
                        if "Remove commas (,)" in clean_option:
                            temp = temp.str.replace(",", "", regex=False)
                        if "Remove currency symbols ($, €, £)" in clean_option:
                            temp = temp.str.replace(r"[€$£]", "", regex=True)
                        if "Remove spaces" in clean_option:
                            temp = temp.str.replace(" ", "", regex=False)
                        df[col] = pd.to_numeric(temp, errors="coerce")

                    elif dtype == "Datetime":
                        df[col] = pd.to_datetime(df[col], format=fmt if fmt else None, errors="coerce")

                    elif dtype == "Categorical":
                        df[col] = df[col].astype("category")

                    st.session_state.df = df
                    add_log(
                        "type_conversion",
                        {"column": col, "from": before_dtype, "to": dtype, "clean_options": clean_option, "datetime_format": fmt},
                        [col]
                    )
                    st.success("Conversion applied successfully.")
                    st.write(f"Before dtype: {before_dtype}")
                    st.write(f"After dtype: {df[col].dtype}")

                except Exception as e:
                    st.error(f"Conversion failed: {e}")

        # ===== MISSING VALUES =====
        with st.expander("Missing Values Handling", expanded=False):
            missing = df.isnull().sum()
            percent = (missing / len(df)) * 100
            missing_summary = pd.DataFrame({"Missing": missing, "%": percent.round(2)})
            st.dataframe(missing_summary, use_container_width=True)

            st.markdown("### Choose missing value action")
            action = st.selectbox(
                "Action",
                [
                    "Drop Rows",
                    "Mean",
                    "Median",
                    "Mode",
                    "Most Frequent",
                    "Constant",
                    "Forward Fill",
                    "Backward Fill"
                ],
                key="missing_action"
            )

            if action == "Drop Rows":
                selected_cols = st.multiselect(
                    "Select column(s) to check for missing values",
                    df.columns.tolist(),
                    key="missing_drop_cols"
                )

                before_rows = len(df)

                if selected_cols:
                    rows_with_missing = df[selected_cols].isnull().any(axis=1).sum()
                    st.write("### Before Preview")
                    st.write(f"Rows before: {before_rows}")
                    st.write(f"Rows that would be removed: {rows_with_missing}")
                    st.write(f"Affected columns: {selected_cols}")

                    if st.button("Apply Missing Handling", key="apply_missing_drop_rows"):
                        if rows_with_missing == 0:
                            st.info("No missing values found in the selected columns.")
                        else:
                            df = df.dropna(subset=selected_cols)
                            after_rows = len(df)

                            st.session_state.df = df
                            add_log(
                                "missing_values_drop_rows",
                                {"columns_checked": selected_cols, "rows_removed": before_rows - after_rows},
                                selected_cols
                            )

                            st.success("Rows with missing values removed.")
                            st.write("### Before vs After")
                            st.write(f"Rows: {before_rows} → {after_rows}")
                            st.write("Affected columns:", selected_cols)

                            missing_new = df.isnull().sum()
                            percent_new = (missing_new / len(df)) * 100
                            st.write("### Updated Missing Values")
                            st.dataframe(
                                pd.DataFrame({"Missing": missing_new, "%": percent_new.round(2)}),
                                use_container_width=True
                            )
                else:
                    st.info("Select one or more columns to enable row dropping.")

            else:
                col = st.selectbox("Select column to handle missing values", df.columns, key="missing_col")
                val = st.text_input("Constant value", key="missing_val") if action == "Constant" else None

                before_rows = len(df)
                before_missing = df[col].isnull().sum()

                st.write("### Before Preview")
                st.write(f"Rows before: {before_rows}")
                st.write(f"Missing in '{col}': {before_missing}")

                if st.button("Apply Missing Handling", key="apply_missing_single_col"):
                    if before_missing == 0:
                        st.info("No missing values in this column. Nothing to handle.")
                    else:
                        try:
                            if action in ["Mean", "Median"] and col not in num_cols:
                                st.error("Mean and Median can only be applied to numeric columns.")

                            elif action == "Mean":
                                df[col] = df[col].fillna(df[col].mean())

                            elif action == "Median":
                                df[col] = df[col].fillna(df[col].median())

                            elif action in ["Mode", "Most Frequent"]:
                                mode_series = df[col].mode(dropna=True)
                                if mode_series.empty:
                                    st.error("No mode is available for this column.")
                                else:
                                    df[col] = df[col].fillna(mode_series.iloc[0])

                            elif action == "Constant":
                                df[col] = df[col].fillna(val)

                            elif action == "Forward Fill":
                                df[col] = df[col].ffill()

                            elif action == "Backward Fill":
                                df[col] = df[col].bfill()

                            if action not in ["Mean", "Median"] or col in num_cols:
                                after_missing = df[col].isnull().sum()

                                st.session_state.df = df
                                add_log(
                                    "missing_values_fill",
                                    {"column": col, "method": action, "constant_value": val if action == "Constant" else None},
                                    [col]
                                )

                                st.success("Missing values handled.")
                                st.write("### Before vs After")
                                st.write(f"Rows: {before_rows} → {len(df)}")
                                st.write(f"Missing in '{col}': {before_missing} → {after_missing}")
                                st.write("Affected columns:", [col])

                                missing_new = df.isnull().sum()
                                percent_new = (missing_new / len(df)) * 100
                                st.write("### Updated Missing Values")
                                st.dataframe(
                                    pd.DataFrame({"Missing": missing_new, "%": percent_new.round(2)}),
                                    use_container_width=True
                                )

                        except Exception as e:
                            st.error(f"Missing value handling failed: {e}")

        # ===== DROP COLUMNS BY MISSING % =====
        with st.expander("Drop Columns by Missing %", expanded=False):
            st.write("### Current Missing %")
            before_missing_pct = (df.isnull().mean() * 100).round(2)
            st.dataframe(before_missing_pct.to_frame(name="% Missing"), use_container_width=True)

            threshold = st.slider("Drop columns above % missing", 0, 100, 50, key="missing_threshold")

            if st.button("Drop Columns", key="drop_cols_threshold_btn"):
                try:
                    before_cols = df.shape[1]
                    to_drop = df.columns[df.isnull().mean() * 100 > threshold].tolist()

                    if len(to_drop) == 0:
                        st.info("No columns exceeded the threshold. Nothing was dropped.")
                    else:
                        df = df.drop(columns=to_drop)
                        after_cols = df.shape[1]

                        st.session_state.df = df
                        add_log(
                            "drop_columns_by_missing_threshold",
                            {"threshold_percent": threshold, "dropped_count": len(to_drop)},
                            to_drop
                        )

                        st.success("Columns dropped.")
                        st.write("### Summary")
                        st.write(f"Columns: {before_cols} → {after_cols}")
                        st.write("Dropped columns:", to_drop)

                        after_missing_pct = (df.isnull().mean() * 100).round(2)
                        st.write("### Remaining Columns")
                        st.dataframe(after_missing_pct.to_frame(name="% Missing"), use_container_width=True)

                except Exception as e:
                    st.error(f"Column drop failed: {e}")

        # ===== DROP SELECTED COLUMNS =====
        with st.expander("Drop Selected Columns", expanded=False):
            cols_to_drop = st.multiselect("Select columns to drop", df.columns, key="drop_selected_cols")

            if st.button("Drop Selected Columns", key="drop_selected_btn"):
                if cols_to_drop:
                    try:
                        before_cols = df.shape[1]
                        df = df.drop(columns=cols_to_drop)
                        after_cols = df.shape[1]

                        st.session_state.df = df
                        add_log(
                            "drop_selected_columns",
                            {"dropped_count": len(cols_to_drop)},
                            cols_to_drop
                        )

                        st.success(f"Dropped {before_cols - after_cols} column(s).")
                        st.write(f"Columns: {before_cols} → {after_cols}")
                        st.write("Dropped:", cols_to_drop)

                    except Exception as e:
                        st.error(f"Could not drop selected columns: {e}")
                else:
                    st.warning("Select at least one column to drop.")

        # ===== DUPLICATES =====
        with st.expander("Duplicates Handling", expanded=False):
            mode = st.radio("Duplicate type", ["Full Row", "Subset"], key="dup_mode")

            if mode == "Subset":
                cols = st.multiselect("Select columns", df.columns, key="dup_cols")
                if not cols:
                    st.warning("Please select at least one column.")
            else:
                cols = None

            if mode == "Full Row":
                dup = df.duplicated()
                dup_all = df[df.duplicated(keep=False)]
            elif mode == "Subset" and cols:
                dup = df.duplicated(subset=cols)
                dup_all = df[df.duplicated(subset=cols, keep=False)]
            else:
                dup = pd.Series([False] * len(df), index=df.index)
                dup_all = pd.DataFrame()

            st.write(f"Duplicates found: {int(dup.sum())}")

            if st.checkbox("Show duplicate rows", key="show_dup_rows"):
                if not dup_all.empty:
                    dup_all = dup_all.copy()
                    if mode == "Full Row":
                        dup_all["Duplicate_Group"] = dup_all.groupby(list(df.columns), dropna=False).ngroup()
                    else:
                        dup_all["Duplicate_Group"] = dup_all.groupby(cols, dropna=False).ngroup()
                    st.dataframe(dup_all.sort_values("Duplicate_Group"), use_container_width=True)
                else:
                    st.write("No duplicates found.")

            keep = st.selectbox("Keep", ["First", "Last"], key="dup_keep")

            if st.button("Remove Duplicates", key="remove_dups_btn"):
                try:
                    before = len(df)

                    if mode == "Full Row":
                        df = df.drop_duplicates(keep="first" if keep == "First" else "last")
                    elif mode == "Subset" and cols:
                        df = df.drop_duplicates(subset=cols, keep="first" if keep == "First" else "last")
                    else:
                        st.warning("Please select subset columns first.")
                        df = df

                    after = len(df)

                    if after != before:
                        st.session_state.df = df
                        add_log(
                            "remove_duplicates",
                            {"mode": mode, "keep": keep, "rows_removed": before - after},
                            cols if cols else list(df.columns)
                        )

                    st.write(f"Rows: {before} → {after}")
                    st.success("Duplicates removed.")

                except Exception as e:
                    st.error(f"Duplicate removal failed: {e}")

        # ===== CATEGORICAL TOOLS =====
        with st.expander("Categorical Tools", expanded=False):
            current_cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

            if len(current_cat_cols) > 0:
                c = st.selectbox("Categorical column", current_cat_cols, key="cat_tool_col")
                std_action = st.selectbox(
                    "Standardize",
                    ["None", "Lower", "Upper", "Title", "Trim"],
                    key="cat_std_action"
                )

                if st.button("Apply Standardization", key="apply_std_btn"):
                    if std_action == "None":
                        st.warning("Please select a standardization option.")
                    else:
                        try:
                            temp_col = df[c].astype("string")

                            if std_action == "Lower":
                                df[c] = temp_col.str.lower()
                            elif std_action == "Upper":
                                df[c] = temp_col.str.upper()
                            elif std_action == "Title":
                                df[c] = temp_col.str.title()
                            elif std_action == "Trim":
                                df[c] = temp_col.str.strip()

                            st.session_state.df = df
                            add_log(
                                "categorical_standardization",
                                {"column": c, "method": std_action},
                                [c]
                            )

                            st.success("Standardization applied.")
                            st.dataframe(df[[c]].head(10), use_container_width=True)

                        except Exception as e:
                            st.error(f"Standardization failed: {e}")

                st.write("### Mapping / Replacement")

                unique_vals = sorted(df[c].dropna().astype(str).str.strip().unique().tolist())
                editor_key = f"mapping_editor_{c}"

                if editor_key not in st.session_state:
                    st.session_state[editor_key] = pd.DataFrame({
                        "old_value": unique_vals,
                        "new_value": unique_vals
                    })

                mapping_df = st.data_editor(
                    st.session_state[editor_key],
                    num_rows="dynamic",
                    use_container_width=True,
                    key=f"mapping_table_{c}"
                )

                set_other = st.checkbox("Set unmatched values to 'Other'", key=f"set_other_{c}")

                if st.button("Apply Mapping", key="apply_mapping_btn"):
                    try:
                        temp_mapping = mapping_df.copy()
                        temp_mapping["old_value"] = temp_mapping["old_value"].astype(str).str.strip()
                        temp_mapping["new_value"] = temp_mapping["new_value"].astype(str).str.strip()

                        temp_mapping = temp_mapping[
                            (temp_mapping["old_value"] != "") &
                            (temp_mapping["old_value"].str.lower() != "nan")
                        ]

                        mapping_dict = dict(zip(temp_mapping["old_value"], temp_mapping["new_value"]))

                        if not mapping_dict:
                            st.warning("Please enter at least one valid mapping row.")
                        else:
                            before_preview = df[[c]].copy()
                            temp_col = df[c].astype(str).str.strip()

                            if set_other:
                                df[c] = temp_col.apply(lambda x: mapping_dict[x] if x in mapping_dict else "Other")
                            else:
                                df[c] = temp_col.replace(mapping_dict)

                            st.session_state.df = df
                            add_log(
                                "categorical_mapping",
                                {"column": c, "set_unmatched_to_other": set_other, "mapping_size": len(mapping_dict)},
                                [c]
                            )

                            st.success("Mapping applied.")
                            st.write("### Before")
                            st.dataframe(before_preview.head(10), use_container_width=True)
                            st.write("### After")
                            st.dataframe(df[[c]].head(10), use_container_width=True)

                    except Exception as e:
                        st.error(f"Mapping failed: {e}")

                thresh = st.slider("Rare category threshold %", 0, 20, 5, key="rare_thresh")
                freq = df[c].value_counts(normalize=True, dropna=False) * 100
                rare = freq[freq < thresh].index.tolist()

                selected = st.multiselect(
                    "Select categories to group into 'Other'",
                    options=rare,
                    key="rare_selected"
                )

                if st.button("Group Rare", key="group_rare_btn"):
                    if len(selected) == 0:
                        st.info("No categories selected.")
                    else:
                        try:
                            before_counts = df[c].value_counts(dropna=False)
                            df[c] = df[c].replace(selected, "Other")
                            after_counts = df[c].value_counts(dropna=False)

                            st.session_state.df = df
                            add_log(
                                "rare_category_grouping",
                                {"column": c, "threshold_percent": thresh, "grouped_categories": selected},
                                [c]
                            )

                            st.success("Rare categories grouped.")
                            st.write("### Before")
                            st.dataframe(before_counts.head(10), use_container_width=True)
                            st.write("### After")
                            st.dataframe(after_counts.head(10), use_container_width=True)

                        except Exception as e:
                            st.error(f"Rare category grouping failed: {e}")

                if st.button("One-hot Encode", key="onehot_btn"):
                    try:
                        original_columns = df.columns.tolist()
                        df = pd.get_dummies(df, columns=[c], dtype=int)
                        new_columns = [col for col in df.columns if col not in original_columns]

                        st.session_state.df = df
                        add_log(
                            "one_hot_encoding",
                            {"original_column": c, "new_columns_created": len(new_columns)},
                            [c] + new_columns
                        )

                        st.success("One-hot encoding applied.")
                        st.dataframe(df.head(10), use_container_width=True)

                    except Exception as e:
                        st.error(f"One-hot encoding failed: {e}")
            else:
                st.info("No categorical columns available.")

        # ===== OUTLIERS =====
        with st.expander("Outlier Detection & Handling", expanded=False):
            current_num_cols = df.select_dtypes(include=np.number).columns.tolist()

            if len(current_num_cols) > 0:
                c = st.selectbox("Select numeric column", current_num_cols, key="outlier_col")
                numeric_series = pd.to_numeric(df[c], errors="coerce")

                if numeric_series.dropna().empty:
                    st.warning("This column has no valid numeric values.")
                else:
                    Q1 = numeric_series.quantile(0.25)
                    Q3 = numeric_series.quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR

                    outliers = df[(numeric_series < lower) | (numeric_series > upper)]

                    st.markdown(f"""
                    <div style='background:#1e1e2e;border-radius:10px;padding:16px;margin-bottom:12px;display:flex;gap:24px'>
                        <div><span style='color:#6060a0;font-size:11px;text-transform:uppercase;letter-spacing:1px'>Outliers</span><br>
                        <span style='color:#f87171;font-family:Space Mono,monospace;font-size:22px;font-weight:700'>{len(outliers):,}</span></div>
                        <div><span style='color:#6060a0;font-size:11px;text-transform:uppercase;letter-spacing:1px'>Lower IQR Bound</span><br>
                        <span style='color:#a78bfa;font-family:Space Mono,monospace;font-size:22px;font-weight:700'>{lower:.2f}</span></div>
                        <div><span style='color:#6060a0;font-size:11px;text-transform:uppercase;letter-spacing:1px'>Upper IQR Bound</span><br>
                        <span style='color:#a78bfa;font-family:Space Mono,monospace;font-size:22px;font-weight:700'>{upper:.2f}</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                    action = st.selectbox(
                        "Choose action",
                        ["Do Nothing", "Remove Rows", "Cap (Winsorize at Quantiles)"],
                        key="outlier_action"
                    )

                    lower_q = None
                    upper_q = None
                    q_option = None

                    if action == "Cap (Winsorize at Quantiles)":
                        q_option = st.selectbox(
                            "Choose quantile cap range",
                            ["1% / 99%", "5% / 95%"],
                            key="winsor_quantile_option"
                        )

                        if q_option == "1% / 99%":
                            lower_q = numeric_series.quantile(0.01)
                            upper_q = numeric_series.quantile(0.99)
                        else:
                            lower_q = numeric_series.quantile(0.05)
                            upper_q = numeric_series.quantile(0.95)

                        st.write(f"Winsorize range: {lower_q:.2f} to {upper_q:.2f}")

                    if st.button("Apply Outlier Handling", key="apply_outlier_btn"):
                        try:
                            before_rows = len(df)

                            if action == "Do Nothing":
                                st.info("No changes applied.")

                            elif len(outliers) == 0 and action == "Remove Rows":
                                st.info("No outliers detected — no rows removed.")

                            elif action == "Remove Rows":
                                df = df[(numeric_series >= lower) & (numeric_series <= upper)]
                                after_rows = len(df)

                                st.session_state.df = df
                                add_log(
                                    "outlier_remove_rows",
                                    {"column": c, "method": "IQR", "rows_removed": before_rows - after_rows},
                                    [c]
                                )

                                st.success("Outlier rows removed.")
                                st.write("### Impact")
                                st.write(f"Outliers detected: {len(outliers)}")
                                st.write(f"Rows: {before_rows} → {after_rows}")

                            elif action == "Cap (Winsorize at Quantiles)":
                                before_capped_low = int((numeric_series < lower_q).sum())
                                before_capped_high = int((numeric_series > upper_q).sum())

                                df[c] = numeric_series.clip(lower=lower_q, upper=upper_q)
                                st.session_state.df = df

                                add_log(
                                    "outlier_winsorization",
                                    {"column": c, "quantile_range": q_option, "values_capped": before_capped_low + before_capped_high},
                                    [c]
                                )

                                st.success("Winsorization applied.")
                                st.write("### Impact")
                                st.write(f"Values capped below lower quantile: {before_capped_low}")
                                st.write(f"Values capped above upper quantile: {before_capped_high}")
                                st.write(f"Total values capped: {before_capped_low + before_capped_high}")
                                st.write(f"Rows: {before_rows} → {before_rows}")

                        except Exception as e:
                            st.error(f"Outlier handling failed: {e}")
            else:
                st.info("No numeric columns available.")

        # ===== SCALING =====
        with st.expander("Scaling", expanded=False):
            current_num_cols = df.select_dtypes(include=np.number).columns.tolist()

            if len(current_num_cols) > 0:
                c = st.multiselect(
                    "Select numeric columns to scale",
                    current_num_cols,
                    key="scale_col"
                )
                m = st.selectbox("Method", ["MinMax", "Z-score"], key="scale_method")

                if st.button("Apply Scaling", key="apply_scaling_btn"):
                    if not c:
                        st.warning("Please select at least one numeric column.")
                    else:
                        try:
                            before_stats = pd.DataFrame({
                                "Column": c,
                                "Mean Before": [pd.to_numeric(df[col], errors="coerce").mean() for col in c],
                                "Std Before": [pd.to_numeric(df[col], errors="coerce").std() for col in c]
                            })

                            scaler = MinMaxScaler() if m == "MinMax" else StandardScaler()
                            df[c] = scaler.fit_transform(df[c])

                            after_stats = pd.DataFrame({
                                "Column": c,
                                "Mean After": [df[col].mean() for col in c],
                                "Std After": [df[col].std() for col in c]
                            })

                            stats_compare = before_stats.merge(after_stats, on="Column")

                            st.session_state.df = df
                            add_log(
                                "scaling",
                                {"columns": c, "method": m},
                                c
                            )

                            st.success("Scaling applied.")
                            st.write("### Before vs After")
                            st.dataframe(stats_compare, use_container_width=True)

                        except Exception as e:
                            st.error(f"Scaling failed: {e}")
            else:
                st.info("No numeric columns available.")

        # ===== COLUMN OPERATIONS =====
        with st.expander("🔧  Column Operations", expanded=False):
            st.write("### Rename Column")
            old_name = st.selectbox("Select column to rename", df.columns, key="rename_col")
            new_name = st.text_input("New column name", key="rename_input")

            if st.button("Rename Column", key="rename_btn"):
                if not new_name.strip():
                    st.warning("Enter a new name.")
                elif new_name in df.columns:
                    st.error("That column name already exists. Choose a different name.")
                else:
                    try:
                        df = df.rename(columns={old_name: new_name})
                        st.session_state.df = df
                        add_log(
                            "rename_column",
                            {"old_name": old_name, "new_name": new_name},
                            [old_name, new_name]
                        )
                        st.success("Column renamed.")

                    except Exception as e:
                        st.error(f"Rename failed: {e}")

            st.markdown("<hr style='border-color:#2a2a3d;margin:20px 0'>", unsafe_allow_html=True)
            st.write("### Create New Column")

            col1 = st.selectbox("Column A", df.columns, key="colA")
            col2 = st.selectbox("Column B", ["None"] + list(df.columns), key="colB")
            operation = st.selectbox(
                "Operation",
                ["Add", "Subtract", "Multiply", "Divide", "Log(A)", "A - Mean(A)"],
                key="operation"
            )
            new_col = st.text_input("New column name", key="create_col_input")

            if st.button("Create Column", key="create_col_btn"):
                if not new_col.strip():
                    st.warning("Enter a new column name.")
                elif new_col in df.columns:
                    st.error("That column name already exists. Choose a different name.")
                else:
                    try:
                        current_num_cols = df.select_dtypes(include=np.number).columns.tolist()

                        if operation in ["Add", "Subtract", "Multiply", "Divide", "Log(A)", "A - Mean(A)"] and col1 not in df.columns:
                            st.error("Column A is invalid.")

                        elif operation in ["Add", "Subtract", "Multiply", "Divide"]:
                            if col2 == "None":
                                st.error("Please select Column B for this operation.")
                            elif col1 not in current_num_cols or col2 not in current_num_cols:
                                st.error("These operations require numeric columns.")
                            else:
                                if operation == "Add":
                                    df[new_col] = df[col1] + df[col2]
                                elif operation == "Subtract":
                                    df[new_col] = df[col1] - df[col2]
                                elif operation == "Multiply":
                                    df[new_col] = df[col1] * df[col2]
                                elif operation == "Divide":
                                    denominator = pd.to_numeric(df[col2], errors="coerce").replace(0, np.nan)
                                    numerator = pd.to_numeric(df[col1], errors="coerce")
                                    df[new_col] = numerator / denominator

                                st.session_state.df = df
                                add_log(
                                    "create_column",
                                    {"new_column": new_col, "operation": operation, "column_a": col1, "column_b": col2},
                                    [col1, col2, new_col]
                                )
                                st.success("New column created.")
                                st.dataframe(df[[new_col]].head(), use_container_width=True)

                        elif operation == "Log(A)":
                            current_num_cols = df.select_dtypes(include=np.number).columns.tolist()
                            if col1 not in current_num_cols:
                                st.error("Log requires a numeric column.")
                            else:
                                safe_series = pd.to_numeric(df[col1], errors="coerce")
                                safe_series = safe_series.where(safe_series > 0, np.nan)
                                df[new_col] = np.log(safe_series)

                                st.session_state.df = df
                                add_log(
                                    "create_column",
                                    {"new_column": new_col, "operation": operation, "column_a": col1},
                                    [col1, new_col]
                                )
                                st.success("New column created.")
                                st.dataframe(df[[new_col]].head(), use_container_width=True)

                        elif operation == "A - Mean(A)":
                            current_num_cols = df.select_dtypes(include=np.number).columns.tolist()
                            if col1 not in current_num_cols:
                                st.error("This operation requires a numeric column.")
                            else:
                                df[new_col] = df[col1] - df[col1].mean()

                                st.session_state.df = df
                                add_log(
                                    "create_column",
                                    {"new_column": new_col, "operation": operation, "column_a": col1},
                                    [col1, new_col]
                                )
                                st.success("New column created.")
                                st.dataframe(df[[new_col]].head(), use_container_width=True)

                    except Exception as e:
                        st.error(f"Column creation failed: {e}")

            st.markdown("<hr style='border-color:#2a2a3d;margin:20px 0'>", unsafe_allow_html=True)
            st.write("### Binning")

            current_num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(current_num_cols) > 0:
                col_bin = st.selectbox("Select numeric column for binning", current_num_cols, key="bin_col")
                bins = st.slider("Number of bins", 2, 10, 4, key="bin_slider")
                method = st.selectbox("Method", ["Equal Width", "Quantile"], key="bin_method")
                bin_col_name = st.text_input("New binned column name", key="bin_input")

                if st.button("Apply Binning", key="apply_binning_btn"):
                    if not bin_col_name.strip():
                        st.warning("Enter a new column name.")
                    elif bin_col_name in df.columns:
                        st.error("That column name already exists. Choose a different name.")
                    else:
                        try:
                            if method == "Equal Width":
                                binned = pd.cut(df[col_bin], bins=bins)
                            else:
                                binned = pd.qcut(df[col_bin], q=bins, duplicates="drop")

                            labels = []
                            for interval in binned.cat.categories:
                                left = round(interval.left, 2)
                                right = round(interval.right, 2)
                                labels.append(f"{left} - {right}")

                            if method == "Equal Width":
                                df[bin_col_name] = pd.cut(df[col_bin], bins=bins, labels=labels)
                            else:
                                df[bin_col_name] = pd.qcut(df[col_bin], q=bins, labels=labels, duplicates="drop")

                            st.session_state.df = df
                            add_log(
                                "binning",
                                {"source_column": col_bin, "new_column": bin_col_name, "bins": bins, "method": method},
                                [col_bin, bin_col_name]
                            )

                            st.success("Binning applied.")
                            st.write("### Preview")
                            st.dataframe(df[[col_bin, bin_col_name]].head(10), use_container_width=True)
                            st.write("### Bin Counts")
                            st.dataframe(df[bin_col_name].value_counts(dropna=False).to_frame("count"), use_container_width=True)

                        except Exception as e:
                            st.error(f"Binning failed: {e}")
            else:
                st.info("No numeric columns available.")

        # ===== DATA VALIDATION RULES =====
        with st.expander("Data Validation Rules", expanded=False):
            validation_type = st.selectbox(
                "Select validation type",
                ["Numeric Range", "Allowed Categories", "Non-null Constraint"],
                key="val_type"
            )

            col_val = st.selectbox("Select column", df.columns, key="val_col")
            violations = pd.DataFrame()

            min_val = None
            max_val = None
            allowed = ""

            if validation_type == "Numeric Range":
                if col_val in df.select_dtypes(include=np.number).columns.tolist():
                    col_min = float(df[col_val].min())
                    col_max = float(df[col_val].max())
                    min_val = st.number_input("Minimum value", value=col_min, key="val_min")
                    max_val = st.number_input("Maximum value", value=col_max, key="val_max")
                else:
                    st.warning("Please select a numeric column for Numeric Range validation.")

            elif validation_type == "Allowed Categories":
                allowed = st.text_input("Enter allowed values (comma-separated)", key="val_allowed")

            check = st.button("Check Violations", key="val_button")

            if check:
                try:
                    if validation_type == "Numeric Range":
                        if col_val not in df.select_dtypes(include=np.number).columns.tolist():
                            st.error("Selected column is not numeric.")
                        elif min_val > max_val:
                            st.error("Minimum cannot be greater than maximum.")
                        else:
                            violations = df[(df[col_val] < min_val) | (df[col_val] > max_val)]

                    elif validation_type == "Allowed Categories":
                        if not allowed.strip():
                            st.error("Please enter at least one allowed value.")
                        else:
                            allowed_list = [x.strip() for x in allowed.split(",") if x.strip()]
                            violations = df[~df[col_val].astype(str).isin(allowed_list)]

                    elif validation_type == "Non-null Constraint":
                        violations = df[df[col_val].isnull()]

                    if not violations.empty:
                        st.error(f"Violations found: {len(violations)}")
                        st.dataframe(violations, use_container_width=True)

                        csv = violations.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download Violations CSV",
                            data=csv,
                            file_name="violations.csv",
                            mime="text/csv",
                            key="val_download"
                        )
                    else:
                        if validation_type == "Allowed Categories" and not allowed.strip():
                            pass
                        else:
                            st.success("No violations found.")

                except Exception as e:
                    st.error(f"Validation check failed: {e}")

        st.markdown("---")
        st.subheader("Transformation Log")

        if st.session_state.log:
            log_df = pd.DataFrame(st.session_state.log).copy()
            log_df["parameters"] = log_df["parameters"].apply(lambda x: str(x))
            log_df["affected_columns"] = log_df["affected_columns"].apply(
                lambda x: ", ".join(map(str, x)) if isinstance(x, list) else str(x)
            )
            st.dataframe(log_df, use_container_width=True)

            if st.button("Undo Last Transformation", key="undo_last_btn"):
                if not st.session_state.log:
                    st.info("Nothing to undo.")
                elif "original_df" not in st.session_state:
                    st.warning("Original dataset not found. Please re-upload.")
                else:
                    st.session_state.log.pop()
                    df_replay = st.session_state.original_df.copy()

                    for step in st.session_state.log:
                        op = step["operation"]
                        p = step["parameters"]
                        try:
                            if op == "type_conversion":
                                col = p["column"]
                                if p["to"] == "Numeric":
                                    df_replay[col] = pd.to_numeric(df_replay[col], errors="coerce")
                                elif p["to"] == "Datetime":
                                    df_replay[col] = pd.to_datetime(df_replay[col], errors="coerce")
                                elif p["to"] == "Categorical":
                                    df_replay[col] = df_replay[col].astype("category")
                            elif op == "missing_values_drop_rows":
                                df_replay = df_replay.dropna(subset=p["columns_checked"])
                            elif op == "missing_values_fill":
                                col = p["column"]
                                method = p["method"]
                                if method == "Mean":
                                    df_replay[col] = df_replay[col].fillna(df_replay[col].mean())
                                elif method == "Median":
                                    df_replay[col] = df_replay[col].fillna(df_replay[col].median())
                                elif method in ["Mode", "Most Frequent"]:
                                    df_replay[col] = df_replay[col].fillna(df_replay[col].mode()[0])
                                elif method == "Constant":
                                    df_replay[col] = df_replay[col].fillna(p["constant_value"])
                                elif method == "Forward Fill":
                                    df_replay[col] = df_replay[col].ffill()
                                elif method == "Backward Fill":
                                    df_replay[col] = df_replay[col].bfill()
                            elif op == "drop_columns_by_missing_threshold":
                                to_drop = df_replay.columns[df_replay.isnull().mean() * 100 > p["threshold_percent"]].tolist()
                                df_replay = df_replay.drop(columns=to_drop)
                            elif op == "drop_selected_columns":
                                cols = step["affected_columns"]
                                existing = [c for c in cols if c in df_replay.columns]
                                df_replay = df_replay.drop(columns=existing)
                            elif op == "remove_duplicates":
                                keep = "first" if p["keep"] == "First" else "last"
                                if p["mode"] == "Full Row":
                                    df_replay = df_replay.drop_duplicates(keep=keep)
                                else:
                                    subset = [c for c in step["affected_columns"] if c in df_replay.columns]
                                    df_replay = df_replay.drop_duplicates(subset=subset, keep=keep)
                            elif op == "scaling":
                                cols = [c for c in p["columns"] if c in df_replay.columns]
                                scaler = MinMaxScaler() if p["method"] == "MinMax" else StandardScaler()
                                df_replay[cols] = scaler.fit_transform(df_replay[cols])
                            elif op == "rename_column":
                                df_replay = df_replay.rename(columns={p["old_name"]: p["new_name"]})
                        except Exception:
                            pass

                    st.session_state.df = df_replay
                    st.success("Last transformation undone.")
                    st.rerun()

            if st.button("Reset All Transformations", key="reset_all_transformations"):
                if "original_df" in st.session_state:
                    st.session_state.df = st.session_state.original_df.copy()
                    st.session_state.log = []
                    st.success("All transformations have been reset.")
                    st.rerun()
                else:
                    st.warning("Original dataset not found. Please re-upload the file.")
        else:
            st.info("No transformations applied yet.")



# ================= PAGE C =================
elif page == "Visualization Builder":

    if "df" not in st.session_state:
        st.warning("Please upload a dataset first")
    else:
        df_vis = st.session_state.df.copy()

        st.header("Visualization Builder")
        st.caption("Build charts from the cleaned dataset.")

        # ---------- detect column types ----------
        num_cols = df_vis.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df_vis.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        datetime_cols = df_vis.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()

        # try to detect date-like object columns
        for col in df_vis.columns:
            if col not in datetime_cols and df_vis[col].dtype == "object":
                parsed = pd.to_datetime(df_vis[col], errors="coerce")
                if parsed.notna().mean() >= 0.8:
                    datetime_cols.append(col)

        # ---------- filters ----------
        st.markdown("---")
        st.subheader("Filters")

        filtered_df = df_vis.copy()

        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:
            cat_filter_col = st.selectbox(
                "Category filter column (optional)",
                ["None"] + cat_cols,
                key="viz_cat_filter_col"
            )

            if cat_filter_col != "None":
                cat_values = filtered_df[cat_filter_col].dropna().astype(str).unique().tolist()
                selected_cat_values = st.multiselect(
                    "Select category values",
                    options=sorted(cat_values),
                    default=sorted(cat_values),
                    key="viz_cat_filter_vals"
                )
                filtered_df = filtered_df[
                    filtered_df[cat_filter_col].astype(str).isin(selected_cat_values)
                ]

        with filter_col2:
            num_filter_col = st.selectbox(
                "Numeric range filter column (optional)",
                ["None"] + num_cols,
                key="viz_num_filter_col"
            )

            if num_filter_col != "None":
                temp_series = pd.to_numeric(filtered_df[num_filter_col], errors="coerce").dropna()

                if not temp_series.empty:
                    min_val = float(temp_series.min())
                    max_val = float(temp_series.max())

                    if min_val == max_val:
                        st.info(f"'{num_filter_col}' has only one value: {min_val}")
                    else:
                        selected_range = st.slider(
                            f"Select range for {num_filter_col}",
                            min_value=min_val,
                            max_value=max_val,
                            value=(min_val, max_val),
                            key="viz_num_range"
                        )
                        filtered_df = filtered_df[
                            pd.to_numeric(filtered_df[num_filter_col], errors="coerce").between(
                                selected_range[0], selected_range[1]
                            )
                        ]
                else:
                    st.info("No valid numeric values available for range filter.")

        st.write(f"Filtered rows: {len(filtered_df)}")

        if filtered_df.empty:
            st.warning("No data left after filtering. Adjust your filters.")
        else:
            # ---------- chart controls ----------
            st.markdown("---")
            st.subheader("Chart Controls")

            plot_type = st.selectbox(
                "Plot type",
                [
                    "Histogram",
                    "Box Plot",
                    "Scatter Plot",
                    "Line Chart",
                    "Grouped Bar Chart",
                    "Correlation Heatmap"
                ],
                key="viz_plot_type"
            )

            aggregation = st.selectbox(
                "Aggregation",
                ["sum", "mean", "count", "median"],
                key="viz_agg"
            )

            top_n = None
            if plot_type == "Grouped Bar Chart":
                top_n = st.slider("Top N categories", 3, 20, 10, key="viz_top_n")

            # ---------- chart rendering ----------
            st.markdown("---")
            st.subheader("Chart Output")

            fig, ax = plt.subplots(figsize=(10, 6))

            try:
                # ===== HISTOGRAM =====
                if plot_type == "Histogram":
                    if not num_cols:
                        st.warning("No numeric columns available for a histogram.")
                    else:
                        hist_col = st.selectbox("Numeric column", num_cols, key="hist_col")
                        hist_data = pd.to_numeric(filtered_df[hist_col], errors="coerce").dropna()

                        if hist_data.empty:
                            st.warning("No valid numeric data available for this histogram.")
                        else:
                            bins = st.slider("Number of bins", 5, 50, 20, key="hist_bins")
                            ax.hist(hist_data, bins=bins, edgecolor="black")
                            ax.set_title(f"Histogram of {hist_col}")
                            ax.set_xlabel(hist_col)
                            ax.set_ylabel("Frequency")
                            st.pyplot(fig)

                # ===== BOX PLOT =====
                elif plot_type == "Box Plot":
                    if not num_cols:
                        st.warning("No numeric columns available for a box plot.")
                    else:
                        y_col = st.selectbox("Numeric column", num_cols, key="box_y")
                        x_col = st.selectbox(
                            "Category column (optional)",
                            ["None"] + cat_cols,
                            key="box_x"
                        )

                        box_data = filtered_df[[y_col]].copy()
                        box_data[y_col] = pd.to_numeric(box_data[y_col], errors="coerce")
                        box_data = box_data.dropna()

                        if box_data.empty:
                            st.warning("No valid numeric data available for this box plot.")
                        else:
                            if x_col == "None":
                                ax.boxplot(box_data[y_col].dropna())
                                ax.set_title(f"Box Plot of {y_col}")
                                ax.set_ylabel(y_col)
                                ax.set_xticklabels([y_col])
                            else:
                                temp = filtered_df[[x_col, y_col]].copy()
                                temp[y_col] = pd.to_numeric(temp[y_col], errors="coerce")
                                temp = temp.dropna()

                                groups = []
                                labels = []
                                for val in temp[x_col].dropna().astype(str).unique():
                                    vals = temp[temp[x_col].astype(str) == val][y_col].dropna()
                                    if not vals.empty:
                                        groups.append(vals)
                                        labels.append(val)

                                if not groups:
                                    st.warning("No valid grouped data available for this box plot.")
                                else:
                                    ax.boxplot(groups, labels=labels)
                                    ax.set_title(f"Box Plot of {y_col} by {x_col}")
                                    ax.set_xlabel(x_col)
                                    ax.set_ylabel(y_col)
                                    plt.xticks(rotation=45)
                                    st.pyplot(fig)

                # ===== SCATTER PLOT =====
                elif plot_type == "Scatter Plot":
                    if len(num_cols) < 2:
                        st.warning("At least two numeric columns are needed for a scatter plot.")
                    else:
                        x_col = st.selectbox("X column", num_cols, key="scatter_x")
                        y_options = [c for c in num_cols if c != x_col]
                        y_col = st.selectbox("Y column", y_options, key="scatter_y")
                        group_col = st.selectbox(
                            "Group/color column (optional)",
                            ["None"] + cat_cols,
                            key="scatter_group"
                        )

                        temp = filtered_df[[x_col, y_col] + ([] if group_col == "None" else [group_col])].copy()
                        temp[x_col] = pd.to_numeric(temp[x_col], errors="coerce")
                        temp[y_col] = pd.to_numeric(temp[y_col], errors="coerce")
                        temp = temp.dropna(subset=[x_col, y_col])

                        if temp.empty:
                            st.warning("No valid numeric pairs available for this scatter plot.")
                        else:
                            if group_col == "None":
                                ax.scatter(temp[x_col], temp[y_col], alpha=0.7)
                            else:
                                for grp in temp[group_col].astype(str).unique():
                                    grp_df = temp[temp[group_col].astype(str) == grp]
                                    ax.scatter(grp_df[x_col], grp_df[y_col], alpha=0.7, label=grp)
                                ax.legend(title=group_col)

                            ax.set_title(f"{y_col} vs {x_col}")
                            ax.set_xlabel(x_col)
                            ax.set_ylabel(y_col)
                            st.pyplot(fig)

                # ===== LINE CHART =====
                elif plot_type == "Line Chart":
                    if not datetime_cols:
                        st.warning("No datetime/date-like columns available for a line chart.")
                    elif not num_cols:
                        st.warning("No numeric columns available for a line chart.")
                    else:
                        x_col = st.selectbox("Date column", datetime_cols, key="line_x")
                        y_col = st.selectbox("Y column", num_cols, key="line_y")
                        group_col = st.selectbox(
                            "Group column (optional)",
                            ["None"] + cat_cols,
                            key="line_group"
                        )

                        temp = filtered_df.copy()
                        temp[x_col] = pd.to_datetime(temp[x_col], errors="coerce")
                        temp[y_col] = pd.to_numeric(temp[y_col], errors="coerce")
                        temp = temp.dropna(subset=[x_col])

                        if temp.empty:
                            st.warning("No valid data available for this line chart.")
                        else:
                            if aggregation == "count":
                                if group_col == "None":
                                    plot_df = temp.groupby(x_col).size().reset_index(name="value")
                                    ax.plot(plot_df[x_col], plot_df["value"])
                                else:
                                    for grp in temp[group_col].astype(str).unique():
                                        grp_df = temp[temp[group_col].astype(str) == grp]
                                        plot_df = grp_df.groupby(x_col).size().reset_index(name="value")
                                        ax.plot(plot_df[x_col], plot_df["value"], label=grp)
                                    ax.legend(title=group_col)
                            else:
                                temp = temp.dropna(subset=[y_col])
                                if temp.empty:
                                    st.warning("No valid numeric values available after cleaning for line chart.")
                                else:
                                    if group_col == "None":
                                        plot_df = temp.groupby(x_col)[y_col].agg(aggregation).reset_index()
                                        ax.plot(plot_df[x_col], plot_df[y_col])
                                    else:
                                        for grp in temp[group_col].astype(str).unique():
                                            grp_df = temp[temp[group_col].astype(str) == grp]
                                            plot_df = grp_df.groupby(x_col)[y_col].agg(aggregation).reset_index()
                                            ax.plot(plot_df[x_col], plot_df[y_col], label=grp)
                                        ax.legend(title=group_col)

                            ax.set_title(f"Line Chart of {y_col} by {x_col}")
                            ax.set_xlabel(x_col)
                            ax.set_ylabel(f"{aggregation} of {y_col}" if aggregation != "count" else "count")
                            plt.xticks(rotation=45)
                            st.pyplot(fig)

                # ===== GROUPED BAR CHART =====
                elif plot_type == "Grouped Bar Chart":
                    if not cat_cols:
                        st.warning("No categorical columns available for a grouped bar chart.")
                    else:
                        x_col = st.selectbox("Category column", cat_cols, key="bar_x")
                        group_col = st.selectbox(
                            "Group column (optional)",
                            ["None"] + [c for c in cat_cols if c != x_col],
                            key="bar_group"
                        )

                        if aggregation == "count":
                            y_col = None
                        else:
                            if not num_cols:
                                st.warning("No numeric columns available for this aggregation.")
                                y_col = None
                            else:
                                y_col = st.selectbox("Y column", num_cols, key="bar_y")

                        temp = filtered_df.copy()

                        if aggregation == "count":
                            if group_col == "None":
                                plot_df = temp[x_col].astype(str).value_counts().head(top_n).reset_index()
                                plot_df.columns = [x_col, "value"]
                                ax.bar(plot_df[x_col], plot_df["value"])
                            else:
                                plot_df = (
                                    temp.groupby([x_col, group_col])
                                    .size()
                                    .reset_index(name="value")
                                )

                                top_categories = (
                                    temp[x_col].astype(str).value_counts().head(top_n).index.tolist()
                                )
                                plot_df = plot_df[plot_df[x_col].astype(str).isin(top_categories)]

                                pivot_df = plot_df.pivot(index=x_col, columns=group_col, values="value").fillna(0)
                                pivot_df.plot(kind="bar", ax=ax)
                        else:
                            if y_col is not None:
                                temp[y_col] = pd.to_numeric(temp[y_col], errors="coerce")
                                temp = temp.dropna(subset=[y_col])

                                summary = temp.groupby(x_col)[y_col].agg(aggregation).sort_values(ascending=False).head(top_n)
                                top_categories = summary.index.tolist()

                                temp = temp[temp[x_col].isin(top_categories)]

                                if group_col == "None":
                                    plot_df = temp.groupby(x_col)[y_col].agg(aggregation).reset_index()
                                    ax.bar(plot_df[x_col], plot_df[y_col])
                                else:
                                    plot_df = temp.groupby([x_col, group_col])[y_col].agg(aggregation).reset_index()
                                    pivot_df = plot_df.pivot(index=x_col, columns=group_col, values=y_col).fillna(0)
                                    pivot_df.plot(kind="bar", ax=ax)

                        ax.set_title("Grouped Bar Chart")
                        ax.set_xlabel(x_col)
                        ax.set_ylabel("count" if aggregation == "count" else f"{aggregation} of {y_col}")
                        plt.xticks(rotation=45)
                        st.pyplot(fig)

                # ===== HEATMAP =====
                elif plot_type == "Correlation Heatmap":
                    if len(num_cols) < 2:
                        st.warning("At least two numeric columns are needed for a correlation heatmap.")
                    else:
                        selected_cols = st.multiselect(
                            "Numeric columns",
                            num_cols,
                            default=num_cols[:min(5, len(num_cols))],
                            key="heatmap_cols"
                        )

                        if len(selected_cols) < 2:
                            st.warning("Select at least two numeric columns.")
                        else:
                            corr = filtered_df[selected_cols].corr(numeric_only=True)

                            im = ax.imshow(corr, cmap="coolwarm", aspect="auto")
                            ax.set_xticks(range(len(corr.columns)))
                            ax.set_yticks(range(len(corr.columns)))
                            ax.set_xticklabels(corr.columns, rotation=45, ha="right")
                            ax.set_yticklabels(corr.columns)
                            ax.set_title("Correlation Heatmap")

                            for i in range(len(corr.index)):
                                for j in range(len(corr.columns)):
                                    ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)

                            fig.colorbar(im, ax=ax)
                            st.pyplot(fig)

            except Exception as e:
                st.error(f"Could not create chart: {e}")


# ================= PAGE D =================
elif page == "Export & Report":

    if "df" not in st.session_state:
        st.warning("Please upload a dataset first on the Upload & Overview page.")
    else:
        df_export = st.session_state.df.copy()
        log_data = st.session_state.get("log", [])

        st.markdown("""
        <p style='font-family:Space Mono,monospace;font-size:11px;color:#4f46e5;letter-spacing:2px;text-transform:uppercase;margin:0 0 8px 0'>Page D</p>
        <h2 style='font-family:Space Mono,monospace;font-size:22px;color:#e2e2f0;margin:0 0 4px 0'>Export & Report</h2>
        <p style='color:#6060a0;font-size:13px;margin:0 0 24px 0'>
            Download your cleaned dataset, transformation report, and reproducible recipe.
        </p>
        """, unsafe_allow_html=True)

        # ---------- helper for Excel export ----------
        def to_excel_bytes(dataframe):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                dataframe.to_excel(writer, index=False, sheet_name="cleaned_data")
            return output.getvalue()

        # ---------- dataset summary ----------
        st.markdown("### Current Dataset Summary")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Rows", df_export.shape[0])
        with c2:
            st.metric("Columns", df_export.shape[1])
        with c3:
            st.metric("Transformations Logged", len(log_data))

        st.caption("Preview of the final cleaned dataset")
        st.dataframe(df_export.head(50), use_container_width=True, height=300)

        st.markdown("---")

        # ---------- export cleaned dataset ----------
        st.subheader("Export Cleaned Dataset")

        csv_data = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Cleaned Dataset (CSV)",
            data=csv_data,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            key="download_cleaned_csv"
        )

        try:
            excel_data = to_excel_bytes(df_export)
            st.download_button(
                label="Download Cleaned Dataset (Excel)",
                data=excel_data,
                file_name="cleaned_dataset.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_cleaned_excel"
            )
        except Exception as e:
            st.error(f"Excel export failed: {e}")

        st.markdown("---")

        # ---------- transformation report ----------
        st.subheader("Transformation Report")

        if log_data:
            report_df = pd.DataFrame(log_data).copy()
            report_df["parameters"] = report_df["parameters"].apply(
                lambda x: json.dumps(x, ensure_ascii=False, indent=2) if isinstance(x, dict) else str(x)
            )
            report_df["affected_columns"] = report_df["affected_columns"].apply(
                lambda x: ", ".join(map(str, x)) if isinstance(x, list) else str(x)
            )

            st.dataframe(report_df, use_container_width=True)

            report_json = {
                "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "final_dataset_shape": {
                    "rows": int(df_export.shape[0]),
                    "columns": int(df_export.shape[1])
                },
                "steps": log_data
            }

            st.download_button(
                label="Download Transformation Report (JSON)",
                data=json.dumps(report_json, indent=2, ensure_ascii=False).encode("utf-8"),
                file_name="transformation_report.json",
                mime="application/json",
                key="download_report_json"
            )

            report_txt_lines = []
            report_txt_lines.append("TRANSFORMATION REPORT")
            report_txt_lines.append(f"Generated at: {report_json['generated_at']}")
            report_txt_lines.append(f"Final dataset shape: {df_export.shape[0]} rows x {df_export.shape[1]} columns")
            report_txt_lines.append("")

            for i, step in enumerate(log_data, start=1):
                report_txt_lines.append(f"Step {i}")
                report_txt_lines.append(f"Timestamp: {step.get('timestamp', '')}")
                report_txt_lines.append(f"Operation: {step.get('operation', '')}")
                report_txt_lines.append(f"Parameters: {step.get('parameters', {})}")
                report_txt_lines.append(f"Affected columns: {step.get('affected_columns', [])}")
                report_txt_lines.append("")

            st.download_button(
                label="Download Transformation Report (TXT)",
                data="\n".join(report_txt_lines).encode("utf-8"),
                file_name="transformation_report.txt",
                mime="text/plain",
                key="download_report_txt"
            )

        else:
            st.info("No transformations have been logged yet. You can still export the cleaned dataset.")
            report_json = {
                "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "final_dataset_shape": {
                    "rows": int(df_export.shape[0]),
                    "columns": int(df_export.shape[1])
                },
                "steps": []
            }

            st.download_button(
                label="Download Empty Transformation Report (JSON)",
                data=json.dumps(report_json, indent=2, ensure_ascii=False).encode("utf-8"),
                file_name="transformation_report.json",
                mime="application/json",
                key="download_empty_report_json"
            )

        st.markdown("---")

        # ---------- reproducible recipe ----------
        st.subheader("Workflow Recipe")

        recipe = {
            "recipe_name": "data_wrangler_recipe",
            "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_dataset_columns": list(df_export.columns),
            "final_dataset_shape": {
                "rows": int(df_export.shape[0]),
                "columns": int(df_export.shape[1])
            },
            "transformation_steps": log_data
        }

        st.code(json.dumps(recipe, indent=2, ensure_ascii=False), language="json")

        st.download_button(
            label="Download Recipe (JSON)",
            data=json.dumps(recipe, indent=2, ensure_ascii=False).encode("utf-8"),
            file_name="workflow_recipe.json",
            mime="application/json",
            key="download_recipe_json"
        )

