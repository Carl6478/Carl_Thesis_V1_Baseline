import streamlit as st
import pandas as pd
import json
import os

LOG_PATH = "experiment_logs.jsonl"


@st.cache_data
def load_logs(path: str):
    if not os.path.exists(path):
        return None

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not rows:
        return None

    df = pd.DataFrame(rows)

    # Only keep rows that have a task_id (HumanEval runs)
    if "task_id" in df.columns:
        df = df[~df["task_id"].isna()]
    else:
        return None

    if df.empty:
        return None

    # Expand metrics dict into columns if present
    if "metrics" in df.columns:
        metrics_df = pd.json_normalize(df["metrics"])
        # prefix with m_ to avoid clashes
        metrics_df.columns = [f"m_{c}" for c in metrics_df.columns]
        df = pd.concat([df.drop(columns=["metrics"]), metrics_df], axis=1)

    return df


st.set_page_config(
    page_title="HumanEval Leaderboard",
    layout="wide",
)

st.title("🏆 HumanEval Leaderboard")
st.markdown(
    "Compare models on logged HumanEval problems using your experiment logs."
)
st.markdown("---")

df = load_logs(LOG_PATH)

if df is None:
    st.warning(
        f"No HumanEval logs found yet.\n\n"
        f"Run some experiments from the main page so that "
        f"`{LOG_PATH}` contains entries with `task_id`."
    )
    st.stop()

# ------------------------------------------------------------------
# Clean up types for safe sorting / filtering
# ------------------------------------------------------------------

# Force model and task_id to string
df["model"] = df["model"].astype(str)
df["task_id"] = df["task_id"].astype(str)

# Drop invalid or empty model entries
df = df[
    (df["model"].notna())
    & (df["model"] != "nan")
    & (df["model"] != "None")
    & (df["model"].str.strip() != "")
]

if df.empty:
    st.warning("All HumanEval log rows had invalid or empty model names.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")

models = sorted(df["model"].unique().tolist())
selected_models = st.sidebar.multiselect(
    "Models", models, default=models
)

tasks = sorted(df["task_id"].unique().tolist())
selected_tasks = st.sidebar.multiselect(
    "HumanEval tasks", tasks, default=tasks
)

df_filtered = df[
    df["model"].isin(selected_models)
    & df["task_id"].isin(selected_tasks)
]

st.markdown("### 📋 Raw HumanEval Logs (filtered)")
st.dataframe(
    df_filtered[
        [
            "timestamp",
            "model",
            "task_id",
            "prompt",
            "raw_output",
        ]
    ],
    use_container_width=True,
)

st.markdown("---")
st.markdown("### 📊 Model-Level Leaderboard")

# Columns that look like metrics (start with m_)
metric_cols = [c for c in df_filtered.columns if c.startswith("m_")]

if not metric_cols:
    st.info(
        "No metric columns found in logs yet (m_*). "
        "Make sure the main app is logging complexity / maintainability metrics."
    )
else:
    # Ensure metrics are numeric where possible
    for c in metric_cols:
        df_filtered[c] = pd.to_numeric(df_filtered[c], errors="coerce")

    agg_dict = {"task_id": "nunique"}
    for c in metric_cols:
        agg_dict[c] = "mean"

    leaderboard = (
        df_filtered.groupby("model")
        .agg(agg_dict)
        .reset_index()
        .rename(columns={"task_id": "n_unique_tasks"})
    )

    # Nice display names for common metrics
    rename_map = {
        "m_cyclomatic_complexity": "avg_cyclomatic",
        "m_cognitive_complexity": "avg_cognitive",
        "m_maintainability_index": "avg_MI",
        "m_loc": "avg_LOC",
        "m_lloc": "avg_LLOC",
        "m_sloc": "avg_SLOC",
        "m_comment_lines": "avg_comments",
        "m_functions": "avg_functions",
        "m_lizard_avg_nloc": "avg_lizard_nloc",
    }
    leaderboard = leaderboard.rename(columns=rename_map)

    st.dataframe(leaderboard, use_container_width=True)

    # Optional: simple bar chart on a key metric (e.g. avg_MI)
    if "avg_MI" in leaderboard.columns:
        st.markdown("#### Average Maintainability Index by Model")
        chart_data = leaderboard.set_index("model")["avg_MI"]
        st.bar_chart(chart_data)

st.markdown("---")
st.caption(
    "This leaderboard is based on logged HumanEval runs in experiment_logs.jsonl. "
    "It summarizes complexity / maintainability metrics per model; "
    "functional correctness (pass@k) still comes from the HumanEval evaluator script."
)
