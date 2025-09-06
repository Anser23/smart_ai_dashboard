# app.py
"""
Smart AI Dashboard - Gradio + Plotly + Groq (fixed)
Replace your existing app.py with this file.
Ensure GROQ_API_KEY is set in Space Settings -> Repository secrets.
"""

import os
import io
import json
import tempfile
import traceback
from datetime import datetime
from typing import Tuple, List, Dict, Optional, Any, Union

import numpy as np
import pandas as pd
import plotly.express as px
import gradio as gr

# ---------------------------
# Utility: safe Groq getter (groq==0.9.0 + httpx 0.27.x)
# ---------------------------
def get_groq_client() -> Tuple[Optional[Any], Optional[str], Optional[str]]:
    """
    Return (client, api_key, error_message_or_None).
    For groq==0.9.0 we use Groq(api_key=...) and do not pass proxies.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None, None, "Missing GROQ_API_KEY (set it in Space → Settings → Repository secrets)."

    try:
        from groq import Groq  # type: ignore
    except Exception as e:
        return None, api_key, f"Groq import failed: {e}"

    try:
        client = Groq(api_key=api_key)
        print("Initialized Groq client via Groq(api_key=...)")
        return client, api_key, None
    except Exception as e:
        # give informative message
        return None, api_key, f"Groq init error: {e}"


# ---------------------------
# Helpers: data IO & cleaning
# ---------------------------
def read_any_table(file_obj) -> pd.DataFrame:
    """Read CSV or Excel (xlsx/xls). Return pandas DataFrame."""
    if file_obj is None:
        raise ValueError("No file provided.")
    name = getattr(file_obj, "name", str(file_obj)).lower()
    try:
        file_obj.seek(0)
    except Exception:
        pass

    if name.endswith(".csv"):
        try:
            df = pd.read_csv(file_obj, sep=None, engine="python")
        except Exception:
            file_obj.seek(0)
            df = pd.read_csv(file_obj)
        return df
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file_obj, engine="openpyxl")
    raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")


def detect_cols(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Detect numeric, categorical, and datetime-like columns (conservative)."""
    if df is None or df.empty:
        return [], [], []
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_like: List[str] = []
    for c in df.columns:
        if c in numeric:
            continue
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            valid_ratio = parsed.notna().mean()
            if valid_ratio >= 0.5:
                datetime_like.append(c)
        except Exception:
            continue
    datetime_like = sorted(list(dict.fromkeys(datetime_like)))
    categorical = [c for c in df.columns if c not in numeric + datetime_like]
    return numeric, categorical, datetime_like


def quick_clean(
    df: pd.DataFrame,
    drop_dupes: bool,
    drop_na: bool,
    fill_method: str,
    fill_value: str
) -> Tuple[pd.DataFrame, Dict]:
    if df is None:
        return df, {"note": "no data"}
    out = df.copy()
    before = len(out)
    if drop_dupes:
        out = out.drop_duplicates()
    if drop_na:
        out = out.dropna()
    else:
        num_cols = out.select_dtypes(include=[np.number]).columns
        if fill_method == "mean":
            for c in num_cols:
                out[c] = out[c].fillna(out[c].mean())
        elif fill_method == "median":
            for c in num_cols:
                out[c] = out[c].fillna(out[c].median())
        elif fill_method == "zero":
            for c in num_cols:
                out[c] = out[c].fillna(0)
        elif fill_method == "value":
            val = fill_value
            try:
                val_cast = float(fill_value)
                val = val_cast
            except Exception:
                pass
            out = out.fillna(val)
    after = len(out)
    summary = {
        "duplicates_removed": int(before - after) if drop_dupes else 0,
        "remaining_nulls": int(out.isna().sum().sum()),
        "rows": int(out.shape[0]),
        "cols": int(out.shape[1]),
    }
    return out, summary


def profile_df(df: pd.DataFrame) -> Dict:
    if df is None:
        return {}
    numeric, categorical, datetime_like = detect_cols(df)
    head_csv = df.head(5).to_csv(index=False)
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "missing_total": int(df.isna().sum().sum()),
        "numeric_cols": numeric,
        "categorical_cols": categorical,
        "datetime_cols": datetime_like,
        "head_rows_csv": head_csv,
    }


# ---------------------------
# Chart builder
# ---------------------------
def build_chart(df: pd.DataFrame, chart_type: str, x: Optional[str], y: Optional[str], color: Optional[str]):
    if df is None or df.empty:
        return None
    color = None if (not color or color == "None") else color
    try:
        if chart_type == "Bar":
            if not x or not y:
                return None
            fig = px.bar(df, x=x, y=y, color=color)
        elif chart_type == "Line":
            if not x or not y:
                return None
            fig = px.line(df, x=x, y=y, color=color)
        elif chart_type == "Scatter":
            if not x or not y:
                return None
            fig = px.scatter(df, x=x, y=y, color=color)
        elif chart_type == "Pie":
            if not x:
                return None
            fig = px.pie(df, names=x, values=(y if y in df.columns else None))
        elif chart_type == "Histogram":
            if not x:
                return None
            fig = px.histogram(df, x=x, nbins=30)
        else:
            return None
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), template="plotly_dark")
        return fig
    except Exception as e:
        print("Chart error:", e)
        return None


def fig_to_png_path(fig) -> Optional[str]:
    if fig is None:
        return None
    try:
        png_bytes = fig.to_image(format="png", scale=2)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(png_bytes)
        tmp.flush()
        tmp.close()
        return tmp.name
    except Exception as e:
        print("fig export error:", e)
        return None


# ---------------------------
# AI Chat handler helpers
# ---------------------------
def parse_ai_chart_spec(text: str):
    try:
        import re
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
        if not m:
            return None
        spec = json.loads(m.group(1))
        return spec.get("chart", None)
    except Exception:
        return None


def normalize_history_to_messages(history: Union[List, Tuple]) -> List[Dict[str, str]]:
    """
    Normalize Gradio Chat history to a list of messages for the LLM:
      - Accepts list of tuples (user, assistant), or list of dicts with keys
      - Returns list of {'role': 'user'|'assistant', 'content': '...'} preserving order
    Always strip extra fields like metadata.
    """
    out: List[Dict[str, str]] = []
    if not history:
        return out

    # history may be a list of (user_text, bot_text) tuples (older gradio)
    # or a list of dicts like {"role":"user","content":"..."} if your UI used that shape.
    for item in history:
        # tuple/list of two items
        if isinstance(item, (list, tuple)) and len(item) == 2:
            u, a = item
            # only add non-empty
            if u is not None and str(u).strip() != "":
                out.append({"role": "user", "content": str(u)})
            if a is not None and str(a).strip() != "":
                out.append({"role": "assistant", "content": str(a)})
            continue
        # dict-like entry possibly from previous code
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            # Some gradio/other clients store content as dict e.g. {"content": [{"type":"text","text":"..."}]}
            if isinstance(content, (list, dict)):
                # try to extract text fields
                if isinstance(content, list):
                    # join text pieces
                    pieces = []
                    for c in content:
                        if isinstance(c, dict) and "text" in c:
                            pieces.append(str(c["text"]))
                        elif isinstance(c, str):
                            pieces.append(c)
                    content_str = " ".join(pieces)
                elif isinstance(content, dict):
                    content_str = content.get("text") or content.get("value") or str(content)
                else:
                    content_str = str(content)
            else:
                content_str = "" if content is None else str(content)

            if role in ("user", "assistant", "system"):
                out.append({"role": role, "content": content_str})
            else:
                # If dict had keys but no role, try to infer: if 'user' in keys or fallback to user
                # But avoid passing metadata or other fields
                if content_str:
                    out.append({"role": "user", "content": content_str})
            continue
        # fallback: item is raw string - treat as user message
        try:
            s = str(item)
            if s.strip():
                out.append({"role": "user", "content": s})
        except Exception:
            continue

    # final cleaning: keep only role and content strings, drop empties
    cleaned: List[Dict[str, str]] = []
    for m in out:
        r = m.get("role", "user")
        c = m.get("content", "")
        if c is None:
            continue
        c = str(c)
        if c.strip() == "":
            continue
        cleaned.append({"role": r, "content": c})
    return cleaned


def sanitize_messages_for_groq(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Ensure each message is a dict with only 'role' and 'content' (string).
    Groq (current SDK) rejects extra fields like 'metadata'.
    """
    out: List[Dict[str, str]] = []
    for m in messages:
        role = m.get("role", "user") if isinstance(m, dict) else "user"
        # role sometimes comes as 'assistant' or 'user' etc.
        if role not in ("system", "user", "assistant"):
            role = "user"
        content = m.get("content") if isinstance(m, dict) else str(m)
        if isinstance(content, (list, dict)):
            # collapse lists/dicts into a text blob if needed
            try:
                content = json.dumps(content)
            except Exception:
                content = str(content)
        if content is None:
            continue
        content_str = str(content).strip()
        if content_str == "":
            continue
        out.append({"role": role, "content": content_str})
    return out


# ---------------------------
# AI Chat handler
# ---------------------------
def chat_with_ai(message: str, history: List[Any], df_state: Optional[pd.DataFrame]):
    """
    Returns: updated_history (list for gradio Chatbot) and optional chart spec (dict)
    This function:
      - normalizes history into messages,
      - sanitizes them to only role+content (no metadata),
      - appends the new user message,
      - calls groq client and returns the assistant string.
    """
    try:
        hist = history or []
        # Normalize existing history into message objects
        msgs_from_history = normalize_history_to_messages(hist)

        # Attach dataset profile if available
        if df_state is not None:
            try:
                prof = profile_df(df_state)
                prof_text = "DATASET PROFILE:\n" + json.dumps(prof, indent=2)
                msgs_from_history.append({"role": "user", "content": prof_text})
            except Exception:
                pass

        # Append current user message
        msgs_from_history.append({"role": "user", "content": str(message)})

        # Sanitize - remove metadata or unsupported fields
        msgs_clean = sanitize_messages_for_groq(msgs_from_history)

        client, key, err = get_groq_client()
        if err:
            reply = f"⚠️ AI init problem: {err}"
            # Return a gradio-compatible update of history: keep previous + assistant reply
            new_hist = hist + [{"role": "user", "content": message}, {"role": "assistant", "content": reply}]
            return new_hist, None

        # Try to call groq - use chat.completions.create (most compatible shape for groq 0.9.0)
        answer = None
        try:
            # Use only role+content strings
            print("Calling Groq with messages count:", len(msgs_clean))
            resp = client.chat.completions.create(
                model="llama3-8b-8192",
                temperature=0.25,
                max_tokens=700,
                messages=msgs_clean,
            )
            # Best-effort parse of response
            try:
                answer = resp.choices[0].message.content.strip()
            except Exception:
                # fallback to str representation
                answer = str(resp)
        except Exception as e:
            # return clear error message rather than raw trace
            tb = traceback.format_exc()
            print("Groq call failed:", e, tb)
            answer = f"⚠️ AI runtime error: {e}"

        # Build gradio chat history representation (append both user and assistant entries)
        new_hist = hist + [{"role": "user", "content": message}, {"role": "assistant", "content": answer}]
        chart_cfg = parse_ai_chart_spec(answer)
        return new_hist, chart_cfg

    except Exception as final_e:
        print("chat_with_ai top-level error:", final_e, traceback.format_exc())
        fallback = history or []
        fallback.append({"role": "assistant", "content": f"Internal error: {final_e}"})
        return fallback, None


# ---------------------------
# CSS & Layout loading (external files or fallback)
# ---------------------------
def load_asset_text(path: str, fallback: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return fallback

FALLBACK_CSS = """
/* minimal fallback CSS if assets/style.css missing */
body { background: #0b1220; color: #93A3B8; font-family: Inter, system-ui, -apple-system, sans-serif; }
.header-hero { padding: 14px 18px; background: linear-gradient(90deg,#071826,#0f172a); color: #E2E8F0; border-bottom: 1px solid #1f2a3f; }
.panel { background:#0f172a; border-radius:12px; padding:12px; border:1px solid #1f2a3f; }
.kpi { padding:10px; border-radius:10px; background: rgba(255,255,255,0.02); }
.gr-button { border-radius:10px !important; }
"""

FALLBACK_HTML = """
<div class="header-hero">
  <h2 style="margin:0;color:#E2E8F0">🤖 Smart AI Dashboard</h2>
  <div style="color:#93A3B8;font-size:0.9rem">Data, charts & AI — portfolio-ready</div>
</div>
"""

STYLE_TEXT = load_asset_text("assets/style.css", FALLBACK_CSS)
LAYOUT_HTML = load_asset_text("assets/layout.html", FALLBACK_HTML)


# ---------------------------
# Build Gradio UI (unchanged structure)
# ---------------------------
with gr.Blocks(css=STYLE_TEXT, theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="emerald")) as demo:
    gr.HTML(LAYOUT_HTML)

    with gr.Row():
        kpi_rows = gr.HTML('<div class="kpi"><b>Rows</b><div id="k_rows">0</div></div>')
        kpi_missing = gr.HTML('<div class="kpi"><b>Missing</b><div id="k_missing">0</div></div>')
        kpi_num = gr.HTML('<div class="kpi"><b>Numeric Cols</b><div id="k_num">0</div></div>')
        kpi_time = gr.HTML('<div class="kpi"><b>Last Updated</b><div id="k_time">—</div></div>')

    with gr.Tabs():
        with gr.TabItem("📥 Data"):
            with gr.Row():
                with gr.Column(scale=1):
                    up = gr.File(label="Upload CSV / Excel (.csv, .xlsx)", file_count="single",
                                 file_types=[".csv", ".xlsx", ".xls"])
                    preview = gr.Dataframe(label="Preview (first 200 rows)", interactive=False)
                    drop_dupes = gr.Checkbox(label="Drop duplicates", value=True)
                    drop_na = gr.Checkbox(label="Drop rows with missing values", value=False)
                    fill_method = gr.Dropdown(["none", "mean", "median", "zero", "value"],
                                              value="none", label="Fill NA method (if not dropping)")
                    fill_value = gr.Textbox(label="Custom fill value", value="0")
                    apply_clean = gr.Button("Apply Cleaning", variant="secondary")
                with gr.Column(scale=1):
                    gr.Markdown("### Chart Builder")
                    chart_type = gr.Dropdown(["Bar", "Line", "Pie", "Scatter", "Histogram"], value="Bar", label="Chart Type")
                    x_col = gr.Dropdown(choices=[], label="X Axis")
                    y_col = gr.Dropdown(choices=[], label="Y Axis (numeric for most charts)")
                    color_col = gr.Dropdown(choices=["None"], value="None", label="Color/Group (optional)")
                    draw = gr.Button("Draw Chart", variant="primary")
                    out_plot = gr.Plot(label="Chart")

        with gr.TabItem("📊 Dashboard"):
            gr.Markdown("**Latest chart & small preview**")
            sample_plot = gr.Plot(label="Latest Chart")
            dash_preview = gr.Dataframe(label="Preview (first 200 rows)", interactive=False)

        with gr.TabItem("💬 AI Assistant"):
            with gr.Row():
                with gr.Column(scale=3):
                    bot = gr.Chatbot(label="Smart Business Assistant", type="messages", height=420, show_copy_button=True)
                    user_msg = gr.Textbox(placeholder="Ask about KPIs, trends, or dataset insights...", label="Message")
                    send = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear Chat")
                with gr.Column(scale=1):
                    hf_status = "✅ GROQ_API_KEY detected" if os.getenv("GROQ_API_KEY") else "⚠️ GROQ_API_KEY missing"
                    gr.Markdown(f"**Model**: Groq LLaMA 3\n\n**Status**: {hf_status}\n\n"
                                "Tip: Upload data in the Data tab for context-aware responses.")
        with gr.TabItem("⚙️ Settings"):
            with gr.Column():
                has_key = bool(os.getenv("GROQ_API_KEY"))
                gr.Markdown("**Secrets / API**")
                gr.Markdown(
                    f"- GROQ_API_KEY detected: {'✅ Yes' if has_key else '⚠️ No'}\n"
                    f"- Add it in your Space: Settings → Repository secrets → GROQ_API_KEY\n"
                )
                gr.Markdown("**About**\nSmart AI Dashboard — lightweight, portfolio-style, Gradio + Plotly + Groq.")

    raw_df_state = gr.State(value=None)
    clean_df_state = gr.State(value=None)
    last_plot_state = gr.State(value=None)

    # ---- Event handlers (same behavior but robust) ----
    def on_upload(file):
        if file is None:
            return None, None, gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=["None"]), \
                   "<div class='kpi'><b>Rows</b><div>0</div></div>", \
                   "<div class='kpi'><b>Missing</b><div>0</div></div>", \
                   "<div class='kpi'><b>Numeric Cols</b><div>0</div></div>", \
                   "<div class='kpi'><b>Last Updated</b><div>—</div></div>", None
        try:
            df = read_any_table(file)
            prev = df.head(200)
            num, cat, dt = detect_cols(df)
            all_for_x = dt + cat + num
            y_choices = num if num else []
            color_choices = ["None"] + cat + num
            k_rows = f"<div class='kpi'><b>Rows</b><div>{df.shape[0]}</div></div>"
            k_missing = f"<div class='kpi'><b>Missing</b><div>{int(df.isna().sum().sum())}</div></div>"
            k_num = f"<div class='kpi'><b>Numeric Cols</b><div>{len(num)}</div></div>"
            k_time = f"<div class='kpi'><b>Last Updated</b><div>{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</div></div>"
            return df, prev, gr.update(choices=all_for_x, value=(all_for_x[0] if all_for_x else None)), \
                   gr.update(choices=y_choices, value=(y_choices[0] if y_choices else None)), \
                   gr.update(choices=color_choices, value="None"), \
                   k_rows, k_missing, k_num, k_time, prev
        except Exception as e:
            print("Upload parse error:", e, traceback.format_exc())
            return None, None, gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=["None"]), \
                   "<div class='kpi'><b>Rows</b><div>0</div></div>", \
                   "<div class='kpi'><b>Missing</b><div>0</div></div>", \
                   "<div class='kpi'><b>Numeric Cols</b><div>0</div></div>", \
                   "<div class='kpi'><b>Last Updated</b><div>—</div></div>", None

    up.change(on_upload, inputs=[up], outputs=[raw_df_state, preview, x_col, y_col, color_col, kpi_rows, kpi_missing, kpi_num, kpi_time, dash_preview])

    def on_clean(raw_df, drop_d, drop_n, method, val):
        if raw_df is None:
            return None, None, "<div class='kpi'><b>Rows</b><div>0</div></div>", \
                   "<div class='kpi'><b>Missing</b><div>0</div></div>", \
                   "<div class='kpi'><b>Numeric Cols</b><div>0</div></div>", \
                   "<div class='kpi'><b>Last Updated</b><div>—</div></div>", None
        try:
            clean_df, summary = quick_clean(raw_df, drop_d, drop_n, method, val)
            prev = clean_df.head(200) if clean_df is not None else None
            k_rows = f"<div class='kpi'><b>Rows</b><div>{summary.get('rows', 0)}</div></div>"
            k_missing = f"<div class='kpi'><b>Missing</b><div>{summary.get('remaining_nulls', 0)}</div></div>"
            num, cat, dt = detect_cols(clean_df) if clean_df is not None else ([], [], [])
            k_num = f"<div class='kpi'><b>Numeric Cols</b><div>{len(num)}</div></div>"
            k_time = f"<div class='kpi'><b>Last Updated</b><div>{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</div></div>"
            return clean_df, prev, k_rows, k_missing, k_num, k_time, prev
        except Exception as e:
            print("Cleaning error:", e, traceback.format_exc())
            return None, None, "<div class='kpi'><b>Rows</b><div>0</div></div>", \
                   "<div class='kpi'><b>Missing</b><div>0</div></div>", \
                   "<div class='kpi'><b>Numeric Cols</b><div>0</div></div>", \
                   "<div class='kpi'><b>Last Updated</b><div>—</div></div>", None

    apply_clean.click(on_clean, inputs=[raw_df_state, drop_dupes, drop_na, fill_method, fill_value],
                      outputs=[clean_df_state, preview, kpi_rows, kpi_missing, kpi_num, kpi_time, dash_preview])

    def on_draw(clean_df, chart_t, x, y, color):
        df_source = clean_df if clean_df is not None else raw_df_state.value
        if df_source is None:
            return None, None, None
        try:
            if y and y in df_source.columns:
                df_source = df_source.copy()
                df_source[y] = pd.to_numeric(df_source[y], errors="coerce")
            fig = build_chart(df_source, chart_t, x, y, color)
            return fig, None, fig
        except Exception as e:
            print("Draw chart error:", e, traceback.format_exc())
            return None, None, None

    draw.click(on_draw, inputs=[clean_df_state, chart_type, x_col, y_col, color_col], outputs=[out_plot, gr.State(value=None), sample_plot])

    # Chat bindings
    def chat_clicked(message, history):
        # prefer cleaned df if available
        df_for_context = None
        try:
            cval = clean_df_state.value
            if cval is not None and not (hasattr(cval, "empty") and cval.empty):
                df_for_context = cval
            else:
                rval = raw_df_state.value
                if rval is not None and not (hasattr(rval, "empty") and rval.empty):
                    df_for_context = rval
        except Exception:
            df_for_context = None

        updated_history, chart_cfg = chat_with_ai(message, history, df_for_context)
        return updated_history, chart_cfg

    send.click(chat_clicked, inputs=[user_msg, bot], outputs=[bot, gr.State(value=None)])
    user_msg.submit(chat_clicked, inputs=[user_msg, bot], outputs=[bot, gr.State(value=None)])
    clear.click(lambda: [], None, bot, queue=False)

# ---------------------------
# Launch
# ---------------------------
if __name__ == "__main__":
    demo.launch(ssr_mode=False)
