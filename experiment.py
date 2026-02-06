import streamlit as st
import subprocess
import pandas as pd
import numpy as np
import ast
import re

from human_eval.data import read_problems

from sentence_transformers import SentenceTransformer, util
from radon.complexity import cc_visit
from radon.metrics import mi_visit
from radon.raw import analyze
import lizard

from cognitive_complexity.api import get_cognitive_complexity
from rouge_score import rouge_scorer
import sacrebleu
from math import comb

import json
from datetime import datetime


# ---------------------------------------------------------
#  Logging
# ---------------------------------------------------------

def save_experiment_log(
    model_name,
    prompt,
    output,
    code,
    metrics,
    text_metrics,
    similarity_score,
    rag_enabled,
    task_id=None,
    pass_1=None,
):
    """Save experiment results to a local JSONL log file."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "task_id": task_id,  # HumanEval task id or None
        "prompt": prompt,
        "rag_enabled": rag_enabled,
        "raw_output": output,
        "extracted_code": code,
        "metrics": metrics,
        "text_overlap": text_metrics,  # may contain keys: "rag", "humaneval"
        "similarity_score": similarity_score,
        "pass_1": pass_1,  # HumanEval pass@1 (0/1 or None)
    }

    log_path = "experiment_logs.jsonl"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------
#  Helper Functions
# ---------------------------------------------------------

def list_ollama_models():
    """Return all locally installed Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        lines = result.stdout.strip().split("\n")[1:]
        models = [line.split()[0] for line in lines if line.strip()]
        return models
    except Exception:
        return []


def run_ollama(model, prompt):
    """Run an inference on a local Ollama model."""
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            text=True,
            encoding="utf-8",
            capture_output=True,
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error running model: {str(e)}"


def extract_python_code(output: str) -> str:
    """Extract Python code from model output, removing explanations/markdown.

    Strategy:
    1) If there's a ```python ... ``` or ``` ... ``` block, use that.
    2) Otherwise, keep lines that look like Python code.
    3) If nothing is found, fall back to the full output.
    """
    if not output:
        return ""

    # 1) Try to extract from fenced code blocks
    fence_match = re.search(
        r"```(?:python)?\s*(.*?)```",
        output,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fence_match:
        code = fence_match.group(1).strip()
        if code:
            return code

    # 2) Fallback: heuristic line filter
    lines = output.splitlines()
    code_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(
            (
                "def ",
                "class ",
                "import ",
                "from ",
                "for ",
                "while ",
                "if ",
                "elif ",
                "else",
                "try:",
                "except",
                "with ",
            )
        ) or line.startswith("    "):
            code_lines.append(line)

    code = "\n".join(code_lines).strip()

    # 3) If still nothing, just return the full output
    return code if code else output.strip()


def analyze_code_complexity(code: str):
    """
    Try to compute as many metrics as possible, even if the code has syntax errors.
    Returns a dict where some values may be None if that metric could not be computed.
    """
    result = {}

    if not code or not code.strip():
        return result  # empty dict, caller will handle

    # Try to build an AST (needed for cognitive complexity)
    try:
        tree = ast.parse(code)
        ast_ok = True
    except SyntaxError:
        tree = None
        ast_ok = False

    # Cyclomatic complexity via radon
    try:
        cc = cc_visit(code)
        result["cyclomatic_complexity"] = (
            sum(x.complexity for x in cc) if cc else 0
        )
    except Exception:
        result["cyclomatic_complexity"] = None

    # Maintainability index via radon
    try:
        mi = mi_visit(code, False)
        result["maintainability_index"] = float(
            np.mean(mi) if isinstance(mi, list) else mi
        )
    except Exception:
        result["maintainability_index"] = None

    # Raw metrics (LOC, etc.) via radon.raw
    try:
        raw = analyze(code)
        result["loc"] = raw.loc
        result["lloc"] = raw.lloc
        result["sloc"] = raw.sloc
        result["comment_lines"] = raw.comments
    except Exception:
        result.setdefault("loc", None)
        result.setdefault("lloc", None)
        result.setdefault("sloc", None)
        result.setdefault("comment_lines", None)

    # Lizard metrics (functions, average nloc)
    try:
        liz = lizard.analyze_file.analyze_source_code("generated", code)
        result["functions"] = len(liz.function_list)
        result["lizard_avg_nloc"] = (
            np.mean([f.nloc for f in liz.function_list])
            if liz.function_list
            else 0
        )
    except Exception:
        result["functions"] = None
        result["lizard_avg_nloc"] = None

    # Cognitive complexity (only if AST was valid)
    try:
        if ast_ok and tree is not None:
            func_nodes = [
                n
                for n in ast.walk(tree)
                if isinstance(
                    n, (ast.FunctionDef, ast.AsyncFunctionDef)
                )
            ]
            if func_nodes:
                cog_values = [
                    get_cognitive_complexity(fn) for fn in func_nodes
                ]
                result["cognitive_complexity"] = float(
                    np.mean(cog_values)
                )
            else:
                result["cognitive_complexity"] = 0.0
        else:
            result["cognitive_complexity"] = None
    except Exception:
        result["cognitive_complexity"] = None

    return result


def compute_text_overlap_metrics(generated: str, reference: str):
    """
    Compute BLEU and ROUGE (1, 2, L) between generated text and a reference.
    Works for code or natural language as plain text.
    """
    generated = (generated or "").strip()
    reference = (reference or "").strip()

    if not generated or not reference:
        return {
            "bleu": None,
            "rouge1": None,
            "rouge2": None,
            "rougeL": None,
        }

    # BLEU via sacrebleu
    bleu_score = sacrebleu.corpus_bleu([generated], [[reference]]).score

    # ROUGE via rouge-score
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    r = scorer.score(reference, generated)

    return {
        "bleu": float(bleu_score),
        "rouge1": float(r["rouge1"].fmeasure),
        "rouge2": float(r["rouge2"].fmeasure),
        "rougeL": float(r["rougeL"].fmeasure),
    }


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator of pass@k for a single problem.

    n: total samples generated
    c: number of correct samples
    k: evaluation budget (k <= n)
    """
    n = int(n)
    c = int(c)
    k = int(k)

    if n <= 0 or k <= 0 or c <= 0 or k > n:
        return 0.0

    return 1.0 - comb(n - c, k) / comb(n, k)


def load_text_from_file(uploaded_file):
    """Extract content from a text or PDF file for RAG."""
    if uploaded_file is None:
        return ""

    name = uploaded_file.name.lower()
    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="replace")
    elif name.endswith(".pdf"):
        try:
            import PyPDF2

            reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for p in reader.pages:
                page_text = p.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        except Exception:
            return "Error reading PDF."
    return ""


@st.cache_data
def load_humaneval_problems():
    """Load HumanEval problems once and cache them."""
    return read_problems()  # returns dict: {task_id: {...}}


# ---------------------------------------------------------
#  Streamlit Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Coding Research Lab",
    layout="wide",
)

st.title("🧪 AI Research Lab — Code Generation, RAG, and Model Evaluation")
st.markdown("Evaluate AI model performance, code quality, and RAG behavior.")
st.markdown("---")

# ---------------------------------------------------------
#  Inputs – Model & Prompt
# ---------------------------------------------------------

col1, col2 = st.columns(2)
with col1:
    models = list_ollama_models()
    selected_model = st.selectbox("Select a Local Ollama Model", models)
with col2:
    enable_rag = st.checkbox("Enable RAG (Attach Reference Document)")

# ---------------------------------------------------------
#  HumanEval Problem Selector (optional)
# ---------------------------------------------------------
st.markdown("### 🧪 HumanEval (optional)")

use_humaneval = st.checkbox(
    "Use HumanEval problem instead of a custom prompt", value=False
)
selected_task_id = None

if use_humaneval:
    problems = load_humaneval_problems()
    task_ids = sorted(problems.keys())

    selected_task_id = st.selectbox("Select HumanEval task ID", task_ids)

    if selected_task_id:
        task = problems[selected_task_id]
        # Show the original HumanEval prompt
        st.markdown(f"**Selected task:** `{selected_task_id}`")
        st.code(task["prompt"], language="python")

        if st.button("📥 Load this HumanEval prompt into the editor"):
            # Store into Streamlit state so the text area uses it
            st.session_state["prompt_text"] = task["prompt"]

# Main prompt editor (linked to session_state)
prompt = st.text_area(
    "Enter Your Prompt", height=180, key="prompt_text"
)

uploaded_file = None
reference_text = ""

if enable_rag:
    uploaded_file = st.file_uploader(
        "Upload PDF or Text File", type=["pdf", "txt"]
    )
    if uploaded_file:
        reference_text = load_text_from_file(uploaded_file)
        st.success("Document Loaded Successfully.")
        st.text_area(
            "Extracted Document Content", reference_text, height=150
        )

# ---------------------------------------------------------
#  Analysis Controls
# ---------------------------------------------------------
st.markdown("### ⚙️ Analysis Controls")
enable_complexity = st.checkbox(
    "Enable complexity metrics (cyclomatic, LOC, functions, etc.)",
    value=True,
)
enable_maintainability = st.checkbox(
    "Enable maintainability metrics (Maintainability Index)",
    value=True,
)
enable_similarity = st.checkbox(
    "Enable similarity metrics (RAG-based semantic similarity + BLEU/ROUGE)",
    value=True,
)

# ---------------------------------------------------------
#  Run Model and Analysis
# ---------------------------------------------------------

if st.button("Run Model"):
    st.markdown("### 🔄 Running Model...")

    # Build final prompt (with or without RAG reference)
    combined_prompt = prompt + (
        "\n\nReference:\n" + reference_text if enable_rag else ""
    )
    output = run_ollama(selected_model, combined_prompt)
    output_code = extract_python_code(output)

    # 🔁 Fallback: if extractor failed but we have some output, treat it as code
    if not output_code.strip() and output and output.strip():
        output_code = output.strip()

    st.markdown("## 🧠 Full Model Output")
    st.code(output, language="python")

    st.markdown("### 🧩 Extracted Code Used for Analysis")
    st.code(output_code or "# (no code extracted)", language="python")

    st.markdown("---")

    # Prepare containers so they always exist
    filtered = {}
    text_metrics = {}   # will hold {"rag": {...}, "humaneval": {...}} if available
    score = None
    pass_1 = None       # HumanEval pass@1

    # ---------------------------------------------------------
    #  Code Metrics (Complexity & Maintainability)
    # ---------------------------------------------------------
    if enable_complexity or enable_maintainability:
        st.markdown("## 📊 Code Analysis Metrics")

        if not output_code.strip():
            st.warning(
                "No code was extracted from the model output, skipping analysis."
            )
        else:
            metrics = analyze_code_complexity(output_code)

            # Build filtered metrics dict based on toggles, skipping None values
            if enable_complexity:
                for key in [
                    "cyclomatic_complexity",
                    "cognitive_complexity",
                    "loc",
                    "lloc",
                    "sloc",
                    "comment_lines",
                    "functions",
                    "lizard_avg_nloc",
                ]:
                    value = metrics.get(key)
                    if value is not None:
                        filtered[key] = value

            if enable_maintainability:
                mi_val = metrics.get("maintainability_index")
                if mi_val is not None:
                    filtered["maintainability_index"] = mi_val

            if filtered:
                st.json(filtered)

                df = pd.DataFrame([filtered])
                st.download_button(
                    label="Download Metrics as CSV",
                    data=df.to_csv(index=False),
                    file_name="metrics.csv",
                    mime="text/csv",
                )
            else:
                st.info(
                    "No metrics could be computed for this output, "
                    "likely because the code is too malformed."
                )

    # ---------------------------------------------------------
    #  Embedding Similarity (RAG Quality) + BLEU/ROUGE vs RAG doc
    # ---------------------------------------------------------
    if enable_similarity and enable_rag and reference_text.strip():
        st.markdown(
            "## 🔍 Semantic Similarity Check with Reference Document"
        )
        model_emb = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        emb_output = model_emb.encode(
            output_code, convert_to_tensor=True
        )
        emb_ref = model_emb.encode(
            reference_text, convert_to_tensor=True
        )
        score = util.pytorch_cos_sim(emb_output, emb_ref).item()
        st.metric(
            "Semantic similarity (0-1)", round(float(score), 4)
        )

        # Text overlap metrics (BLEU / ROUGE) between generated code and reference document
        rag_metrics = compute_text_overlap_metrics(
            output_code, reference_text
        )
        text_metrics["rag"] = rag_metrics

        st.markdown("### 🧪 Text Overlap Metrics vs RAG Document (BLEU / ROUGE)")
        st.json(
            {
                "BLEU": None
                if rag_metrics["bleu"] is None
                else round(rag_metrics["bleu"], 4),
                "ROUGE-1 (F1)": None
                if rag_metrics["rouge1"] is None
                else round(rag_metrics["rouge1"], 4),
                "ROUGE-2 (F1)": None
                if rag_metrics["rouge2"] is None
                else round(rag_metrics["rouge2"], 4),
                "ROUGE-L (F1)": None
                if rag_metrics["rougeL"] is None
                else round(rag_metrics["rougeL"], 4),
            }
        )

    # ---------------------------------------------------------
    #  BLEU/ROUGE vs HumanEval Reference Solution
    # ---------------------------------------------------------
    if use_humaneval and selected_task_id is not None:
        problems = load_humaneval_problems()
        task = problems.get(selected_task_id)

        if task is not None and "canonical_solution" in task:
            ref_code = task["canonical_solution"]

            st.markdown("## 🧪 BLEU / ROUGE vs HumanEval Reference Solution")
            st.markdown("**HumanEval canonical solution:**")
            st.code(ref_code, language="python")

            he_metrics = compute_text_overlap_metrics(output_code, ref_code)
            text_metrics["humaneval"] = he_metrics

            st.json(
                {
                    "BLEU": None
                    if he_metrics["bleu"] is None
                    else round(he_metrics["bleu"], 4),
                    "ROUGE-1 (F1)": None
                    if he_metrics["rouge1"] is None
                    else round(he_metrics["rouge1"], 4),
                    "ROUGE-2 (F1)": None
                    if he_metrics["rouge2"] is None
                    else round(he_metrics["rouge2"], 4),
                    "ROUGE-L (F1)": None
                    if he_metrics["rougeL"] is None
                    else round(he_metrics["rougeL"], 4),
                }
            )
        else:
            st.info(
                "No canonical solution found for this HumanEval task; "
                "skipping BLEU/ROUGE vs reference."
            )

    # ---------------------------------------------------------
    #  HumanEval pass@1 functional correctness (via check_correctness)
    # ---------------------------------------------------------
    if use_humaneval and selected_task_id is not None and output_code.strip():
        try:
            from human_eval.execution import check_correctness

            problems = load_humaneval_problems()
            task = problems.get(selected_task_id)

            if task is not None:
                # WARNING: this executes untrusted model code – keep this in a safe env.
                result = check_correctness(
                    problem=task,
                    completion=output_code,
                    timeout=5,  # seconds
                )
                # result is a dict with at least a "passed" key
                passed = bool(result.get("passed", False))
                pass_1 = 1 if passed else 0

                st.markdown("## 🎯 HumanEval pass@1 Score")
                st.metric("pass@1", pass_1)

                # Also add pass_1 into metrics dict if you want it in the JSON view
                filtered["pass_1"] = pass_1

        except Exception as e:
            st.error(f"Error computing pass@1: {e}")
            pass_1 = None

    # ---------------------------------------------------------
    #  Save results (always inside the Run Model block)
    # ---------------------------------------------------------
    save_experiment_log(
        model_name=selected_model,
        prompt=combined_prompt,
        output=output,
        code=output_code,
        metrics=filtered,
        text_metrics=text_metrics,
        similarity_score=score,
        rag_enabled=enable_rag,
        task_id=selected_task_id,
        pass_1=pass_1,
    )

    st.success("📁 Results saved to experiment_logs.jsonl")


# ---------------------------------------------------------
#  pass@k Helper (HumanEval-style)
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### 🎯 pass@k Helper (HumanEval-style)")

with st.expander("Compute pass@k from your experiment logs"):
    n_samples = st.number_input(
        "Total samples n", min_value=0, step=1, value=0
    )
    n_correct = st.number_input(
        "Number of correct samples c", min_value=0, step=1, value=0
    )
    k_value = st.number_input(
        "k (solutions budget)", min_value=1, step=1, value=1
    )

    if st.button("Compute pass@k", key="compute_passk"):
        pk = estimate_pass_at_k(n_samples, n_correct, k_value)
        st.metric("Estimated pass@k", f"{pk:.4f}")
        st.caption("Formula: 1 - C(n - c, k) / C(n, k)")


# ---------------------------------------------------------
#  Debug Console
# ---------------------------------------------------------
st.sidebar.markdown("---")
if st.sidebar.checkbox("Show Debug Console"):
    st.sidebar.subheader("Session State")
    st.sidebar.json(
        {k: str(v)[:500] for k, v in st.session_state.items()}
    )

# ---------------------------------------------------------
#  Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    """
### 📘 Research Framework Notes
Supports PhD-level evaluation of:
- RAG vs. non-RAG code generation  
- Code complexity & maintainability  
- Semantic similarity between output & reference material  
- Model behavior evaluation across domains  

**For academic citation:**

Hijazi, H. (2025). *Human-Centered Engineering Education–AI Reasoning (HEAR) Framework for Code Generation*. MSc Thesis.
"""
)
st.markdown(
    "###### © 2025 — AI RAG Research Framework for Code Generation Studies"
)
