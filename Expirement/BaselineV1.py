import streamlit as st
import subprocess
import pandas as pd
import numpy as np
import ast
import re
import json
import os
import multiprocessing as mp
import traceback
import uuid
from datetime import datetime
from math import comb
from io import BytesIO
from supabase import create_client, Client

# Set multiprocessing start method (critical for Streamlit + Windows)
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # Already set, ignore


# Create results directory on startup
os.makedirs("results", exist_ok=True)


# ---------------------------------------------------------
#  Optional Imports (graceful)
# ---------------------------------------------------------
def optional_import(module_name: str, attr: str = None):
    try:
        mod = __import__(module_name, fromlist=[attr] if attr else [])
        return getattr(mod, attr) if attr else mod, None
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------
#  Supabase client (for logging to your Supabase project)
# ---------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
_supabase_client: Client | None = None


def get_supabase_client() -> Client | None:
    """
    Returns a cached Supabase client, or None if not configured.
    Uses SUPABASE_URL and SUPABASE_KEY environment variables.
    """
    global _supabase_client

    if _supabase_client is not None:
        return _supabase_client

    if not SUPABASE_URL or not SUPABASE_KEY:
        return None

    try:
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        _supabase_client = None

    return _supabase_client


human_eval_data, human_eval_data_err = optional_import("human_eval.data", "read_problems")
human_eval_exec, human_eval_exec_err = optional_import("human_eval.execution", "check_correctness")


radon_cc, radon_cc_err = optional_import("radon.complexity", "cc_visit")
radon_halstead, radon_halstead_err = optional_import("radon.metrics", "h_visit")  # Halstead
# NOTE: We no longer need mi_visit/raw/lizard for your requested metrics


cog_api, cog_err = optional_import("cognitive_complexity.api", "get_cognitive_complexity")


openpyxl_mod, openpyxl_err = optional_import("openpyxl")


st_mod, st_err = optional_import("sentence_transformers", "SentenceTransformer")
util_mod, util_err = optional_import("sentence_transformers", "util")


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
    run_id=None,
    attempt_index=None,
):
    # Generate a unique ID for this log entry (for easy referencing later)
    log_id = uuid.uuid4().hex

    entry = {
        "log_id": log_id,
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "attempt_index": attempt_index,
        "model": model_name,
        "task_id": task_id,
        "prompt": prompt,
        "rag_enabled": rag_enabled,
        "raw_output": output,
        "extracted_code": code,
        "metrics": metrics,
        "text_overlap": text_metrics,
        "similarity_score": similarity_score,
        "pass_1": pass_1,
    }

    # Existing local log (kept as-is)
    with open("experiment_logs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    # New: send to Supabase 
    try:
        sb = get_supabase_client()
        if sb:
            # Ensure JSON-serializable object for PostgREST
            sb_entry = {
                "log_id": log_id,
                "timestamp": entry["timestamp"],
                "run_id": entry["run_id"],
                "attempt_index": entry["attempt_index"],
                "model": entry["model"],
                "task_id": entry["task_id"],
                "prompt": entry["prompt"],
                "rag_enabled": entry["rag_enabled"],
                "raw_output": entry["raw_output"],
                "extracted_code": entry["extracted_code"],
                "metrics": entry["metrics"],
                "text_overlap": entry["text_overlap"],
                "similarity_score": entry["similarity_score"],
                "pass_1": entry["pass_1"],
            }
            sb.table("experiment_logs").insert(sb_entry).execute()
    except Exception:
        # Do not break the Streamlit app if Supabase insert fails
        traceback.print_exc()


def save_single_run_log(
    model_name,
    prompt,
    output,
    code,
    metrics,
    similarity_score,
    rag_enabled,
    use_humaneval=False,
    selected_task_id=None,
    task_prompt=None,
    canonical_solution=None,
):
    """
    Separate logger for the simple one-off "Run Model" path.
    Writes to a dedicated Supabase table: single_runs.
    Each entry also gets a globally unique run_id for easier referencing.
    """
    # Short, readable run ID (e.g. run_2f4a9b1c)
    run_id = "run_" + uuid.uuid4().hex[:8]

    entry = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "prompt": prompt,
        "rag_enabled": rag_enabled,
        "raw_output": output,
        "extracted_code": code,
        "metrics": metrics,
        "similarity_score": similarity_score,
        "use_humaneval": use_humaneval,
        "task_id": selected_task_id,
        "task_prompt": task_prompt,
        "canonical_solution": canonical_solution,
    }

    # Local backup log file (optional, separate from experiment_logs.jsonl)
    try:
        with open("single_run_logs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        traceback.print_exc()

    # Send to Supabase (single_runs table)
    try:
        sb = get_supabase_client()
        if sb:
            sb.table("single_runs").insert(entry).execute()
    except Exception:
        traceback.print_exc()


def save_pass_at_k_run(
    model_name,
    task_id,
    task_prompt,
    canonical_solution,
    use_humaneval,
    rag_enabled,
    pass_1,
    pass_k,
    k,
    n_attempts,
    c_passed,
    attempts,
    raw_output_attempt1,
    extracted_code_attempt1,
    metrics_attempt1,
    similarity_score=None,
):
    """
    Save a pass@k run to Supabase (table: pass_at_k_runs) and optionally to local JSONL.
    So you can retrieve and view Full Model Output, Generated Code, Canonical, Task Prompt,
    Metrics, and all attempts from the Results page.
    """
    # Short, readable run ID (e.g. run_2f4a9b1c)
    run_id = "run_" + uuid.uuid4().hex[:8]
    # Build JSON-serializable attempts (Supabase jsonb)
    attempts_data = []
    for a in attempts:
        attempts_data.append({
            "attempt": a.get("attempt"),
            "output": a.get("output") or "",
            "code": a.get("code") or "",
            "passed": a.get("passed"),
            "metrics": a.get("metrics") or {},
        })
    entry = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "task_id": task_id,
        "task_prompt": task_prompt,
        "canonical_solution": canonical_solution,
        "use_humaneval": use_humaneval,
        "rag_enabled": rag_enabled,
        "pass_1": pass_1,
        "pass_k": pass_k,
        "k": k,
        "n_attempts": n_attempts,
        "c_passed": c_passed,
        "attempts": attempts_data,
        "raw_output": raw_output_attempt1,
        "extracted_code": extracted_code_attempt1,
        "metrics": metrics_attempt1,
        "similarity_score": similarity_score,
    }
    try:
        with open("pass_at_k_run_logs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        traceback.print_exc()
    try:
        sb = get_supabase_client()
        if sb:
            sb.table("pass_at_k_runs").insert(entry).execute()
    except Exception:
        traceback.print_exc()


# ---------------------------------------------------------
#  Ollama helpers
# ---------------------------------------------------------
def list_ollama_models():
    try:
        r = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if r.returncode != 0:
            return [], r.stderr.strip() or "ollama list failed"
        lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
        if len(lines) <= 1:
            return [], "No models found (ollama list is empty)"
        models = []
        for ln in lines[1:]:
            parts = ln.split()
            if parts:
                models.append(parts[0])
        return models, None
    except Exception as e:
        return [], str(e)


def run_ollama(model, prompt):
    try:
        r = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            text=True,
            encoding="utf-8",
            capture_output=True,
        )
        if r.returncode != 0:
            return f"Error running model:\n{r.stderr.strip()}"
        return r.stdout.strip()
    except Exception as e:
        return f"Error running model: {str(e)}"


# ---------------------------------------------------------
#  Text/code helpers
# ---------------------------------------------------------
def extract_python_code(output: str) -> str:
    """
    Tries to extract python code from fenced blocks first.
    If none, falls back to heuristic extraction.
    """
    if not output:
        return ""


    fence = re.search(r"```(?:python)?\s*(.*?)```", output, flags=re.I | re.S)
    if fence and fence.group(1).strip():
        return fence.group(1).strip()


    lines = output.splitlines()
    code_lines = []
    for line in lines:
        s = line.strip()
        if s.startswith(
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
                "return ",
            )
        ):
            code_lines.append(line)
        elif line.startswith("    ") or line.startswith("\t"):
            code_lines.append(line)


    code = "\n".join(code_lines).strip()
    return code if code else output.strip()


# ---------------------------------------------------------
#  HumanEval execution with multiprocessing timeout (Windows-safe, no signals)
#  IMPORTANT: This custom implementation bypasses human_eval.execution.check_correctness
#  because that function uses signal.setitimer which is not available on Windows.
#  Instead, we directly execute the test code and enforce timeout via process management.
# ---------------------------------------------------------
def _unsafe_execute_worker(task, completion, out_q):
    """
    Executes the HumanEval evaluation in a child process.
    Windows-compatible version that doesn't use signal.setitimer.
    Timeout is enforced by parent process via process.join().
    """
    try:
        # Build the test program (prompt + completion + test)
        check_program = (
            task["prompt"] + completion + "\n" +
            task["test"] + "\n" +
            f"check({task['entry_point']})"
        )
        
        # Debug: Include the full program in output for debugging
        debug_program = check_program[:500]  # First 500 chars
        
        # Execute in isolated namespace
        exec_globals = {}
        try:
            exec(check_program, exec_globals)
            # If we get here, all tests passed
            out_q.put({
                "passed": True,
                "result": "passed",
                "task_id": task["task_id"],
            })
        except AssertionError as e:
            # Test failed (wrong output)
            error_msg = str(e) if str(e) else "assertion failed"
            out_q.put({
                "passed": False,
                "result": f"failed: AssertionError: {error_msg}",
                "task_id": task["task_id"],
                "completion": completion[:200],  # First 200 chars for debugging
                "debug_program": debug_program,  # What was actually executed
            })
        except SyntaxError as e:
            # Syntax error in generated code
            out_q.put({
                "passed": False,
                "result": f"failed: SyntaxError: {e}",
                "task_id": task["task_id"],
                "completion": completion[:200],  # First 200 chars for debugging
                "error_line": e.lineno,
                "debug_program": debug_program,  # What was actually executed
            })
        except Exception as e:
            # Other error (NameError, TypeError, etc.)
            out_q.put({
                "passed": False,
                "result": f"failed: {type(e).__name__}: {e}",
                "task_id": task["task_id"],
                "completion": completion[:200],  # First 200 chars for debugging
                "debug_program": debug_program,  # What was actually executed
            })
    except Exception as e:
        # Catch-all for any unexpected errors
        out_q.put({
            "passed": False,
            "result": f"failed: {e}",
            "traceback": traceback.format_exc(),
        })


def check_correctness_windows(task, completion, timeout_s=30.0):
    """
    Windows-safe correctness check (no signal.setitimer).
    Enforces timeout by killing the process.
    """
    # Check for empty completion
    if not completion or not completion.strip():
        return {
            "passed": False,
            "result": "failed: empty completion",
            "task_id": task.get("task_id", "unknown")
        }
    
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_unsafe_execute_worker, args=(task, completion, q))
    p.daemon = False
    p.start()
    p.join(timeout_s)


    if p.is_alive():
        p.terminate()
        p.join(1.0)
        if p.is_alive():
            p.kill()
            p.join(0.5)
        return {"passed": False, "result": "timeout"}


    if not q.empty():
        return q.get()


    return {"passed": False, "result": "failed: worker produced no result", "exit_code": p.exitcode}


def normalize_humaneval_completion(prompt: str, completion: str) -> str:
    """
    Normalize HumanEval completion code.
    HumanEval expects just the completion (function body), not a redefined function.
    CRITICAL: Must preserve indentation for the function body.
    """
    completion = completion.strip()
    if not completion:
        return completion
   
    # Remove function definition if present (starts with "def ")
    lines = completion.splitlines()
    if lines and lines[0].strip().startswith("def "):
        # Find the function body (after the def line)
        body_lines = []
        in_function = False
        for line in lines:
            if line.strip().startswith("def "):
                in_function = True
                continue
            if in_function:
                # Check if this is the start of another top-level definition
                stripped = line.strip()
                if stripped and not line.startswith((" ", "\t")) and (
                    stripped.startswith("def ") or
                    stripped.startswith("class ") or
                    stripped.startswith("@")
                ):
                    break
                body_lines.append(line)
        
        # Join body lines - DO NOT use .strip() as it removes leading indentation!
        # The body needs to maintain its indentation structure
        if body_lines:
            # Remove trailing empty lines only
            while body_lines and not body_lines[-1].strip():
                body_lines.pop()
            
            # Ensure minimum indentation is at least 4 spaces
            min_indent = float('inf')
            for line in body_lines:
                if line.strip():  # Skip empty lines
                    indent = len(line) - len(line.lstrip())
                    min_indent = min(min_indent, indent)
            
            # If minimum indentation is 0, add 4 spaces to all non-empty lines
            if min_indent == 0 or min_indent == float('inf'):
                body_lines = ["    " + line if line.strip() else line for line in body_lines]
            
            completion = "\n".join(body_lines)
        else:
            completion = ""
   
    return completion


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Estimate Pass@k using the unbiased estimator formula.
    
    Formula: Pass@k = 1 - C(n-c, k) / C(n, k)
    
    Parameters:
    - n: Total number of code samples/attempts generated
    - c: Number of correct samples (that passed all tests)
    - k: Number of samples to consider (e.g., k=1 for Pass@1, k=2 for Pass@2)
    
    Returns:
    - Float between 0.0 and 1.0 representing the estimated probability
      that at least one of k samples will be correct
    
    Example:
    - If n=2 attempts, c=1 correct, k=2: Pass@2 = 1 - C(1,2)/C(2,2) = 1 - 0/1 = 1.0
    - If n=2 attempts, c=0 correct, k=2: Pass@2 = 1 - C(2,2)/C(2,2) = 1 - 1/1 = 0.0
    - If n=2 attempts, c=2 correct, k=2: Pass@2 = 1 - C(0,2)/C(2,2) = 1.0 (can't choose 2 from 0)
    """
    n, c, k = int(n), int(c), int(k)
    
    # Validation: ensure parameters are valid
    if n <= 0 or k <= 0 or k > n:
        return 0.0
    
    # If no correct samples, Pass@k = 0
    if c <= 0:
        return 0.0
    
    # If all samples are correct, Pass@k = 1
    if c >= n:
        return 1.0
    
    # If k > (n - c), we can't choose k items from (n-c) failures
    # This means we're guaranteed to get at least one correct answer
    if k > (n - c):
        return 1.0
    
    # Apply the formula: 1 - C(n-c, k) / C(n, k)
    return 1.0 - comb(n - c, k) / comb(n, k)


def load_text_from_file(uploaded_file):
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="replace")
    if name.endswith(".pdf"):
        pdf_mod, pdf_err = optional_import("PyPDF2")
        if not pdf_mod:
            return f"PyPDF2 not installed: {pdf_err}"
        try:
            reader = pdf_mod.PdfReader(uploaded_file)
            out = []
            for p in reader.pages:
                t = p.extract_text()
                if t:
                    out.append(t)
            return "\n".join(out)
        except Exception as e:
            return f"Error reading PDF: {e}"
    return ""


# ---------------------------------------------------------
#  Metrics (ONLY requested)
# ---------------------------------------------------------
def analyze_code_metrics(code: str, metrics_to_compute=None, code_for_loc=None):
    """
    Computes:
      - Lines of Code (LOC) - excluding comments
      - Comment count - number of lines that are comments (start with #)
      - Cyclomatic Complexity (CC) via radon.cc_visit
      - Cognitive Complexity via cognitive_complexity
      - Halstead Effort via radon.metrics.h_visit

    metrics_to_compute:
      - Optional iterable of metric keys to compute
        (e.g. {"lines_of_code", "cyclomatic_complexity"}).
      - If None, all metrics are computed (default behavior).
    
    code_for_loc:
      - Optional separate code string to use for LOC, comment count, and all complexity
        metrics (CC, cognitive, Halstead). If None, uses the main `code` parameter.
      - Use this so metrics reflect generated code only (e.g. exclude prompt stubs).
    """
    result = {
        "lines_of_code": None,
        "comment_count": None,
        "cyclomatic_complexity": None,
        "cognitive_complexity": None,
        "halstead_effort": None,
    }

    # Normalize metrics_to_compute to a set of internal keys
    if metrics_to_compute is not None:
        metrics_to_compute = set(metrics_to_compute)

    if not code or not code.strip():
        return result

    # Use generated-code-only when provided (for LOC and all complexity metrics)
    loc_code = code_for_loc if code_for_loc is not None else code
    complexity_code = code_for_loc if code_for_loc is not None else code

    # Lines of Code (excluding comments) and Comment count
    if metrics_to_compute is None or any(m in metrics_to_compute for m in ["lines_of_code", "comment_count"]):
        try:
            raw_lines = [line.strip() for line in loc_code.splitlines() if line.strip()]
            
            # Identify which lines are comments (start with #)
            comment_lines = [l for l in raw_lines if l.startswith('#')]
            # Identify which lines are actual code (not starting with #)
            code_only_lines = [l for l in raw_lines if not l.startswith('#')]
            
            # Calculate Lines of Code (excluding comments)
            if metrics_to_compute is None or "lines_of_code" in metrics_to_compute:
                result["lines_of_code"] = len(code_only_lines)
                
            # Comment count: number of lines that are comments
            if metrics_to_compute is None or "comment_count" in metrics_to_compute:
                result["comment_count"] = len(comment_lines)
                    
        except Exception:
            result["lines_of_code"] = None
            result["comment_count"] = None


    # Cyclomatic Complexity (Radon) — on generated code only when code_for_loc provided
    if (metrics_to_compute is None or "cyclomatic_complexity" in metrics_to_compute) and radon_cc:
        try:
            cc_blocks = radon_cc(complexity_code)
            result["cyclomatic_complexity"] = sum(x.complexity for x in cc_blocks) if cc_blocks else 0
        except Exception:
            result["cyclomatic_complexity"] = None


    # Cognitive Complexity (requires AST + functions) — on generated code only when code_for_loc provided
    if metrics_to_compute is None or "cognitive_complexity" in metrics_to_compute:
        try:
            tree = ast.parse(complexity_code)
            ast_ok = True
        except SyntaxError:
            tree = None
            ast_ok = False


        if cog_api and ast_ok and tree is not None:
            try:
                funcs = [
                    n for n in ast.walk(tree)
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                if funcs:
                    vals = [cog_api(fn) for fn in funcs]
                    result["cognitive_complexity"] = float(np.mean(vals))
                else:
                    result["cognitive_complexity"] = 0.0
            except Exception:
                result["cognitive_complexity"] = None


    # Halstead Effort (Radon) — on generated code only when code_for_loc provided
    if (metrics_to_compute is None or "halstead_effort" in metrics_to_compute) and radon_halstead:
        try:
            h = radon_halstead(complexity_code)


            effort = None


            # Most common: h.total.effort
            if hasattr(h, "total") and hasattr(h.total, "effort"):
                effort = h.total.effort


            # Dict-style: h["total"].effort
            elif isinstance(h, dict) and "total" in h and hasattr(h["total"], "effort"):
                effort = h["total"].effort


            # Rare: nested dict: h["total"]["effort"]
            elif isinstance(h, dict) and "total" in h and isinstance(h["total"], dict):
                effort = h["total"].get("effort")


            result["halstead_effort"] = None if effort is None else float(effort)


        except Exception:
            result["halstead_effort"] = None
    else:
        result["halstead_effort"] = None


    return result


# ---------------------------------------------------------
#  Batch Processing
# ---------------------------------------------------------
def run_batch_evaluation(
    models,
    task_ids,
    problems,
    compute_metrics=True,
    metrics_to_compute=None,
    attempts_per_task=2,
    progress_bar=None,
    status_text=None,
):
    """
    Run batch evaluation: for each (model, task_id), generate N attempts and evaluate.
    Returns list of aggregated rows for CSV.
    """
    # Create readable timestamp for run_id and filenames
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_id = f"batch_{timestamp}"
    rows = []

    # Safety: clamp attempts to a sensible range (1–10)
    if attempts_per_task < 1:
        attempts_per_task = 1
    if attempts_per_task > 10:
        attempts_per_task = 10

    total_jobs = len(models) * len(task_ids)
    current_job = 0

    for model in models:
        for task_id in task_ids:
            current_job += 1
            if progress_bar:
                progress_bar.progress(current_job / total_jobs)
            if status_text:
                status_text.text(f"Processing: {model} × {task_id} ({current_job}/{total_jobs})")

            task = problems.get(task_id)
            if not task:
                continue

            prompt = task.get("prompt", "")
            if not prompt:
                continue

            # Per-attempt tracking
            attempts_passed = []
            attempts_eval = []
            attempts_metrics = []

            # Run N attempts
            for attempt_index in range(1, attempts_per_task + 1):
                attempt_output = ""
                attempt_code = ""
                attempt_eval = None
                attempt_passed = None
                attempt_metrics = {}

                try:
                    attempt_output = run_ollama(model, prompt)
                    attempt_code = extract_python_code(attempt_output)
                    if not attempt_code.strip() and attempt_output.strip():
                        attempt_code = attempt_output.strip()

                    # Evaluate
                    completion = None
                    eval_skipped_reason = None
                    if not human_eval_exec:
                        eval_skipped_reason = "human_eval_exec is None"
                    elif not attempt_code.strip():
                        eval_skipped_reason = "attempt_code is empty"

                    if human_eval_exec and attempt_code.strip():
                        try:
                            completion = normalize_humaneval_completion(prompt, attempt_code)
                            r = check_correctness_windows(task, completion, timeout_s=30.0)
                            attempt_eval = r

                            debug_info = {
                                "result_type": type(r).__name__,
                                "result_keys": list(r.keys()) if isinstance(r, dict) else "not_dict",
                                "result_repr": str(r)[:200] if not isinstance(r, dict) else None,
                            }

                            attempt_passed = None
                            if isinstance(r, dict):
                                if "passed" in r:
                                    passed_value = r["passed"]
                                    debug_info["passed_value"] = passed_value
                                    debug_info["passed_type"] = type(passed_value).__name__
                                    if isinstance(passed_value, bool):
                                        attempt_passed = passed_value
                                    elif isinstance(passed_value, str):
                                        attempt_passed = passed_value.lower() in ("true", "passed", "1", "yes")
                                    elif passed_value is None:
                                        attempt_passed = None
                                    else:
                                        attempt_passed = bool(passed_value)
                                elif "result" in r:
                                    result_value = r["result"]
                                    debug_info["result_value"] = result_value
                                    attempt_passed = (result_value == "passed" or result_value is True)
                                else:
                                    debug_info["no_passed_or_result"] = True
                                    attempt_passed = False
                            else:
                                debug_info["not_dict"] = True
                                attempt_passed = False

                            if isinstance(attempt_eval, dict):
                                attempt_eval["_debug"] = debug_info
                            else:
                                attempt_eval = {"result": str(attempt_eval), "_debug": debug_info}
                        except Exception as e:
                            attempt_eval = {"exception": str(e), "exception_type": type(e).__name__}
                            attempt_passed = None

                    # Metrics (only if passed)
                    if compute_metrics and attempt_code.strip() and attempt_passed is True:
                        code_for_metrics = prompt + "\n" + attempt_code
                        attempt_metrics = analyze_code_metrics(code_for_metrics, metrics_to_compute=metrics_to_compute, code_for_loc=attempt_code)

                    # Log attempt
                    text_metrics = {}
                    if attempt_eval:
                        text_metrics["humaneval_result"] = attempt_eval
                        if isinstance(attempt_eval, dict) and "result" in attempt_eval:
                            text_metrics["humaneval_result_string"] = attempt_eval["result"]
                    if completion is not None:
                        text_metrics["normalized_completion"] = completion
                    if eval_skipped_reason:
                        text_metrics["eval_skipped"] = eval_skipped_reason
                    text_metrics["_debug_attempt_passed"] = attempt_passed

                    save_experiment_log(
                        model_name=model,
                        prompt=prompt,
                        output=attempt_output,
                        code=attempt_code,
                        metrics=attempt_metrics,
                        text_metrics=text_metrics,
                        similarity_score=None,
                        rag_enabled=False,
                        task_id=task_id,
                        pass_1=1 if attempt_passed else (0 if attempt_passed is False else None),
                        run_id=run_id,
                        attempt_index=attempt_index,
                    )
                except Exception as e:
                    error_metrics = {"error": str(e)}
                    if attempt_eval:
                        error_metrics["humaneval_result"] = attempt_eval
                    save_experiment_log(
                        model_name=model,
                        prompt=prompt,
                        output=attempt_output,
                        code=attempt_code,
                        metrics=attempt_metrics,
                        text_metrics=error_metrics,
                        similarity_score=None,
                        rag_enabled=False,
                        task_id=task_id,
                        pass_1=None,
                        run_id=run_id,
                        attempt_index=attempt_index,
                    )

                attempts_passed.append(attempt_passed)
                attempts_eval.append(attempt_eval)
                attempts_metrics.append(attempt_metrics)

            # Aggregate metrics using Pass@k
            n = attempts_per_task
            c = sum(1 for p in attempts_passed if p is True)

            # Pass@1 from first attempt
            if attempts_passed:
                if attempts_passed[0] is True:
                    pass_1 = 1
                elif attempts_passed[0] is False:
                    pass_1 = 0
                else:
                    pass_1 = None
            else:
                pass_1 = None

            # Pass@k with k = attempts_per_task
            if not attempts_passed or all(p is None for p in attempts_passed):
                pass_k = None
            else:
                pass_k = estimate_pass_at_k(n=n, c=c, k=attempts_per_task)

            # Backwards-compatible debug fields for first two attempts
            attempt1_passed = attempts_passed[0] if len(attempts_passed) >= 1 else None
            attempt2_passed = attempts_passed[1] if len(attempts_passed) >= 2 else None
            attempt1_eval = attempts_eval[0] if len(attempts_eval) >= 1 else None
            attempt2_eval = attempts_eval[1] if len(attempts_eval) >= 2 else None
            attempt1_metrics = attempts_metrics[0] if len(attempts_metrics) >= 1 else {}
            attempt2_metrics = attempts_metrics[1] if len(attempts_metrics) >= 2 else {}

            attempt1_result_str = None
            attempt2_result_str = None
            if attempt1_eval:
                if isinstance(attempt1_eval, dict):
                    attempt1_result_str = attempt1_eval.get("result", "unknown")
                else:
                    attempt1_result_str = str(attempt1_eval)
            if attempt2_eval:
                if isinstance(attempt2_eval, dict):
                    attempt2_result_str = attempt2_eval.get("result", "unknown")
                else:
                    attempt2_result_str = str(attempt2_eval)

            row = {
                "result_id": uuid.uuid4().hex,
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "task_id": task_id,
                "k": attempts_per_task,
                "n_total_attempts": n,
                "c_correct_attempts": c,
                "pass_1": pass_1,
                "pass_k": pass_k,
                "pass_k_2": pass_k,
                "attempt1_passed": 1 if attempt1_passed is True else (0 if attempt1_passed is False else None),
                "attempt2_passed": 1 if attempt2_passed is True else (0 if attempt2_passed is False else None),
                "attempt1_result": attempt1_result_str,
                "attempt2_result": attempt2_result_str,
                "attempt1_passed_raw": attempt1_passed,
                "attempt2_passed_raw": attempt2_passed,
                "attempt1_lines_of_code": attempt1_metrics.get("lines_of_code"),
                "attempt1_comment_count": attempt1_metrics.get("comment_count"),
                "attempt1_cyclomatic_complexity": attempt1_metrics.get("cyclomatic_complexity"),
                "attempt1_cognitive_complexity": attempt1_metrics.get("cognitive_complexity"),
                "attempt1_halstead_effort": attempt1_metrics.get("halstead_effort"),
                "attempt2_lines_of_code": attempt2_metrics.get("lines_of_code"),
                "attempt2_comment_count": attempt2_metrics.get("comment_count"),
                "attempt2_cyclomatic_complexity": attempt2_metrics.get("cyclomatic_complexity"),
                "attempt2_cognitive_complexity": attempt2_metrics.get("cognitive_complexity"),
                "attempt2_halstead_effort": attempt2_metrics.get("halstead_effort"),
            }
            rows.append(row)
   
    # Save to Excel (new file per run)
    excel_path = None
    if rows:
        df = pd.DataFrame(rows)

        # Best-effort: push batch summary rows to Supabase
        try:
            sb = get_supabase_client()
            if sb:
                sb.table("batch_results").insert(rows).execute()
        except Exception:
            traceback.print_exc()

        # Create descriptive filename with model count, task count, and readable timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        n_models = len(models)
        n_tasks = len(task_ids)
        excel_path = f"results/batch_{n_models}models_{n_tasks}tasks_{timestamp}.xlsx"
       
        # Create sheet name (shorter version for Excel's 31 char limit)
        sheet_name = f"{n_models}m_{n_tasks}t_{timestamp}"[:31]
       
        # Create new Excel file for this run
        if openpyxl_mod:
            try:
                with pd.ExcelWriter(excel_path, engine='openpyxl', mode='w') as writer:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            except Exception as e:
                st.error(f"Failed to save Excel file: {e}")
        else:
            # Fallback to CSV if openpyxl not available
            csv_path = f"results/batch_{n_models}models_{n_tasks}tasks_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            excel_path = csv_path  # Return CSV path if Excel not available
   
    return rows, run_id, excel_path


# ---------------------------------------------------------
#  Cached model loaders
# ---------------------------------------------------------
@st.cache_data
def load_humaneval_problems():
    if not human_eval_data:
        return None
    return human_eval_data()


@st.cache_resource
def get_embedder():
    if not st_mod:
        return None
    return st_mod("sentence-transformers/all-MiniLM-L6-v2")


# ---------------------------------------------------------
#  UI
# ---------------------------------------------------------
st.set_page_config(page_title="AI Coding Research Lab (v2)", layout="wide")
st.title("🧪 AI Research Lab — Code Generation, RAG, and Evaluation (v2)")
st.caption("Graceful deps • cached embeddings • optional HumanEval")


with st.sidebar:
    st.subheader("Navigation")
    page = st.radio(
        "Page",
        ["Run / Configure", "Results"],
        index=0,
    )


# ---------------------------------------------------------
#  Results Page (Supabase-backed)
# ---------------------------------------------------------
if page == "Results":
    st.markdown("## 📊 Saved Results from Supabase")

    sb = get_supabase_client()
    if not sb:
        st.error("Supabase client is not configured. Check SUPABASE_URL / SUPABASE_KEY.")
        st.stop()

    table = st.selectbox(
        "Select results table",
        ["single_runs", "pass_at_k_runs", "experiment_logs", "batch_results"],
        index=0,
    )
    limit = st.number_input("Max rows to load", min_value=1, max_value=50, value=10, step=1)

    st.markdown("### 🔎 Filters")
    c1, c2, c3 = st.columns(3)
    with c1:
        # Build a dropdown of available model names from the selected table
        model_filter = None
        model_options = ["All models"]
        try:
            # Best-effort: fetch recent model values and de-duplicate client-side
            models_res = (
                sb.table(table)
                .select("model")
                .order("timestamp", desc=True)
                .limit(500)
                .execute()
            )
            models_rows = getattr(models_res, "data", None)
            if models_rows is None and isinstance(models_res, dict):
                models_rows = models_res.get("data")
            if models_rows:
                seen = set()
                for r in models_rows:
                    m = r.get("model")
                    if m and m not in seen:
                        seen.add(m)
                model_options = ["All models"] + sorted(seen)
        except Exception:
            # If this fails (no timestamp column, table empty, etc.) we still render the UI
            model_options = ["All models"]

        selected_model_filter = st.selectbox("Model", options=model_options, index=0)
    with c2:
        sort_order = st.selectbox("Sort order", ["Newest first", "Oldest first"], index=0)
    with c3:
        rag_filter = None
        if table in ("single_runs", "pass_at_k_runs", "experiment_logs"):
            rag_filter = st.selectbox("RAG", ["All", "On", "Off"], index=0)
        else:
            st.caption("RAG filter not available for this table.")

    search_term = st.text_input(
        "Search",
        placeholder="Search run_id, task_id, model...",
        key="results_search",
    ).strip()

    try:
        query = sb.table(table).select("*")
        # Most tables have a timestamp column; order desc if present
        if table in ("single_runs", "pass_at_k_runs", "experiment_logs", "batch_results"):
            query = query.order("timestamp", desc=(sort_order == "Newest first"))

        # Apply filters (best-effort; skips when not applicable)
        if selected_model_filter and selected_model_filter != "All models":
            query = query.eq("model", selected_model_filter)
        if rag_filter and rag_filter != "All" and table in ("single_runs", "pass_at_k_runs", "experiment_logs"):
            query = query.eq("rag_enabled", rag_filter == "On")

        res = query.limit(int(limit)).execute()

        rows = getattr(res, "data", None)
        if rows is None and isinstance(res, dict):
            rows = res.get("data")

        # Client-side search across run_id, task_id, model (and id/log_id/result_id if present)
        if rows and search_term:
            term_lower = search_term.lower()
            searchable_keys = ["run_id", "task_id", "model", "log_id", "result_id"]
            filtered = []
            for r in rows:
                for key in searchable_keys:
                    val = r.get(key)
                    if val is not None and term_lower in str(val).lower():
                        filtered.append(r)
                        break
            rows = filtered

        if not rows:
            if search_term:
                st.info("No rows match your search. Try a different term or clear the search.")
            else:
                st.info("No rows found in this table yet.")
        else:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

            # Optional: per-row detailed view similar to main page layout
            if st.checkbox("Show detailed expanders for each row"):
                st.markdown("### 🔍 Row Details")
                for i, row in enumerate(rows):
                    header = f"{i+1}. {row.get('model', '(no model)')} @ {row.get('timestamp', '')}"
                    with st.expander(header, expanded=False):
                        # Common metadata
                        st.markdown("**Metadata**")
                        st.write({
                            "id": row.get("id"),
                            "model": row.get("model"),
                            "timestamp": row.get("timestamp"),
                            "rag_enabled": row.get("rag_enabled"),
                            "task_id": row.get("task_id"),
                        })

                        # Single simple run: show generated vs canonical vs prompt like main page
                        if table == "single_runs":
                            gen_code = row.get("extracted_code") or "# (no code extracted)"
                            canonical = row.get("canonical_solution")
                            task_prompt_val = row.get("task_prompt") or row.get("prompt") or ""

                            col_gen, col_can = st.columns(2)
                            with col_gen:
                                st.markdown("#### 🧩 Generated Code")
                                st.code(gen_code, language="python")
                            with col_can:
                                st.markdown("#### 📌 Canonical Solution")
                                if canonical:
                                    st.code(canonical, language="python")
                                else:
                                    st.info("No canonical solution available for this row.")

                            if task_prompt_val:
                                st.markdown("#### 📜 Task Prompt")
                                st.code(task_prompt_val, language="python")

                            metrics_val = row.get("metrics")
                            if metrics_val:
                                st.markdown("#### 📊 Metrics")
                                st.json(metrics_val)

                        # Experiment logs: show prompt, extracted code, metrics
                        elif table == "experiment_logs":
                            st.markdown("#### 📝 Prompt")
                            st.code(row.get("prompt") or "", language="python")

                            st.markdown("#### 🧩 Extracted Code")
                            st.code(row.get("extracted_code") or "# (no code extracted)", language="python")

                            if row.get("metrics"):
                                st.markdown("#### 📊 Metrics")
                                st.json(row.get("metrics"))

                            if row.get("text_overlap"):
                                st.markdown("#### 🧪 HumanEval / Text Metrics")
                                st.json(row.get("text_overlap"))

                        # Pass@k runs: same layout as run page (full output, generated vs canonical, metrics, attempts)
                        elif table == "pass_at_k_runs":
                            st.markdown("#### 🧠 Full Model Output (Attempt 1)")
                            st.code(row.get("raw_output") or "# (no output)", language="python")
                            col_gen, col_can = st.columns(2)
                            with col_gen:
                                st.markdown("#### 🧩 Generated Code (Attempt 1)")
                                st.code(row.get("extracted_code") or "# (no code extracted)", language="python")
                            with col_can:
                                st.markdown("#### 📌 Canonical Solution")
                                canonical = row.get("canonical_solution")
                                if canonical:
                                    st.code(canonical, language="python")
                                else:
                                    st.info("No canonical solution available.")
                            task_prompt_val = row.get("task_prompt")
                            if task_prompt_val:
                                st.markdown("#### 📜 Task Prompt")
                                st.code(task_prompt_val, language="python")
                            if row.get("metrics"):
                                st.markdown("#### 📊 Code Metrics (Attempt 1)")
                                st.json(row.get("metrics"))
                            st.markdown("#### 📐 Pass@k Summary")
                            st.json({
                                "pass_1": row.get("pass_1"),
                                "pass_k": row.get("pass_k"),
                                "k": row.get("k"),
                                "n_attempts": row.get("n_attempts"),
                                "c_passed": row.get("c_passed"),
                            })
                            if row.get("similarity_score") is not None:
                                st.metric("Semantic similarity (0–1)", round(float(row["similarity_score"]), 4))
                            attempts_data = row.get("attempts") or []
                            if attempts_data:
                                st.markdown("#### 🔄 Attempts")
                                for idx, att in enumerate(attempts_data):
                                    lab = att.get("attempt", idx + 1)
                                    status = "✅ PASSED" if att.get("passed") else ("❌ FAILED" if att.get("passed") is False else "⚠️ ERROR")
                                    with st.expander(f"Attempt {lab} — {status}", expanded=False):
                                        st.markdown("**Raw output**")
                                        st.code(att.get("output") or "", language="text")
                                        st.markdown("**Extracted code**")
                                        st.code(att.get("code") or "# (none)", language="python")
                                        if att.get("metrics"):
                                            st.json(att["metrics"])
                        # Batch results: show summary metrics
                        elif table == "batch_results":
                            st.markdown("#### 📊 Batch Summary Fields")
                            st.json({
                                "k": row.get("k"),
                                "n_total_attempts": row.get("n_total_attempts"),
                                "c_correct_attempts": row.get("c_correct_attempts"),
                                "pass_1": row.get("pass_1"),
                                "pass_k": row.get("pass_k"),
                                "attempt1_result": row.get("attempt1_result"),
                                "attempt2_result": row.get("attempt2_result"),
                            })

    except Exception as e:
        st.error(f"Failed to load results from Supabase: {e}")
        st.stop()

    # Stop here so the rest of the app UI doesn't render on the Results page
    st.stop()


st.markdown("---")


col1, col2 = st.columns(2)
with col1:
    models, ollama_err = list_ollama_models()
    if ollama_err:
        st.error(f"Ollama issue: {ollama_err}")
    selected_model = st.selectbox("Select a Local Ollama Model", models if models else ["(none found)"])
with col2:
    enable_rag = st.checkbox("Enable RAG (attach reference doc)")


st.markdown("### 🧪 HumanEval (optional)")
use_humaneval = st.checkbox("Use HumanEval problem instead of custom prompt", value=False)
selected_task_id = None


if use_humaneval:
    problems = load_humaneval_problems()
    if not problems:
        st.warning("HumanEval not available (install human-eval). Custom prompt still works.")
        use_humaneval = False
    else:
        task_ids = sorted(problems.keys())
        selected_task_id = st.selectbox("Select HumanEval task ID", task_ids)
        if selected_task_id:
            task = problems[selected_task_id]
            col_p, col_c = st.columns(2)
            with col_p:
                st.markdown("**Task Prompt**")
                st.code(task.get("prompt", ""), language="python")
            with col_c:
                st.markdown("**Canonical Solution**")
                if "canonical_solution" in task and task["canonical_solution"]:
                    st.code(task["canonical_solution"], language="python")
                else:
                    st.info("No canonical solution available for this task.")
            if st.button("📥 Load this HumanEval prompt into editor"):
                st.session_state["prompt_text"] = task["prompt"]


prompt = st.text_area("Enter your prompt", height=180, key="prompt_text")


reference_text = ""
if enable_rag:
    uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    if uploaded_file:
        reference_text = load_text_from_file(uploaded_file)
        st.success("Document loaded")
        st.text_area("Extracted document content", reference_text, height=150)


st.markdown("### ⚙️ Analysis Controls")
enable_metrics = st.checkbox("Enable LOC + Comment count + CC + Cognitive + Halstead Effort", value=True)

metrics_to_compute = set()
if enable_metrics:
    st.markdown("**Select which metrics to compute:**")
    metric_label_to_key = {
        "Lines of Code (LOC)": "lines_of_code",
        "Comment count": "comment_count",
        "Cyclomatic Complexity (CC)": "cyclomatic_complexity",
        "Cognitive Complexity": "cognitive_complexity",
        "Halstead Effort": "halstead_effort",
    }
    default_metric_labels = [
        "Lines of Code (LOC)",
        "Comment count",
        "Cyclomatic Complexity (CC)",
        "Cognitive Complexity",
        "Halstead Effort",
    ]
    selected_metric_labels = st.multiselect(
        "Metrics",
        options=list(metric_label_to_key.keys()),
        default=default_metric_labels,
        key="single_run_metric_list",
    )
    metrics_to_compute = {metric_label_to_key[label] for label in selected_metric_labels}
    
    if not metrics_to_compute:
        st.warning("No metrics selected — metrics will be skipped.")

enable_similarity = st.checkbox("Enable similarity", value=True)


# Pass@k option (only available when using HumanEval)
enable_pass_at_k = False
pass_at_k_value = 1
if use_humaneval and human_eval_exec:
    enable_pass_at_k = st.checkbox("Enable pass@k evaluation", value=False)
    if enable_pass_at_k:
        pass_at_k_value = st.number_input("Number of attempts (k)", min_value=1, max_value=10, value=2, step=1)


st.markdown("---")


if st.button("Run Model", type="primary", disabled=(not models)):
    if not models:
        st.stop()


    combined_prompt = prompt + ("\n\nReference:\n" + reference_text if enable_rag else "")
   
    # Handle pass@k evaluation
    if enable_pass_at_k and use_humaneval and selected_task_id and human_eval_exec:
        problems = load_humaneval_problems() or {}
        task = problems.get(selected_task_id)
        if not task:
            st.error("Task not found.")
            st.stop()
       
        task_prompt = task.get("prompt", "")
        attempts = []
        passed_attempts = []
       
        st.markdown(f"## 🔄 Running up to {pass_at_k_value} attempt(s) for pass@{pass_at_k_value} (stop when pass)")
        progress_bar = st.progress(0)
       
        for i in range(pass_at_k_value):
            progress_bar.progress((i + 1) / pass_at_k_value)
            st.markdown(f"### Attempt {i+1}/{pass_at_k_value}")
           
            # Run model
            output = run_ollama(selected_model, task_prompt)
            
            # Show full raw output in expander
            with st.expander("📝 Full Raw Model Output", expanded=False):
                st.code(output, language="text")
            
            # Extract code
            output_code = extract_python_code(output)
            if not output_code.strip() and output.strip():
                output_code = output.strip()
            
            # Show extracted code
            if output_code.strip():
                st.markdown("**✂️ Extracted Code:**")
                st.code(output_code, language="python")
            else:
                st.warning("⚠️ No code was extracted from model output!")
           
            # Evaluate
            attempt_passed = None
            completion = None
            eval_result = None
            if output_code.strip() and human_eval_exec:
                try:
                    # Normalize completion (HumanEval expects completion, not redefined def)
                    completion = normalize_humaneval_completion(task_prompt, output_code)
                    
                    # Use Windows-safe multiprocessing-based timeout wrapper (no signals)
                    result = check_correctness_windows(task, completion, timeout_s=30.0)
                    eval_result = result
                    
                    # Display result
                    st.write("**HumanEval result:**")
                    st.json(result)
                    
                    # Show what was actually executed if there's a failure
                    if not attempt_passed and "debug_program" in result:
                        with st.expander("🐛 Debug: Full Test Program (what was executed)", expanded=False):
                            st.code(result["debug_program"], language="python")
                    
                    # Handle different result structures
                    if isinstance(result, dict):
                        if "passed" in result:
                            attempt_passed = bool(result["passed"])
                        elif "result" in result:
                            attempt_passed = (result["result"] == "passed")
                        else:
                            attempt_passed = False
                    else:
                        attempt_passed = False
                    if attempt_passed:
                        passed_attempts.append(i + 1)
                except Exception as e:
                    attempt_passed = None
                    st.error(f"Evaluation exception: {e}")
           
            # Compute metrics (only if attempt passed)
            # Note: We only compute metrics on working code to avoid misleading measurements
            attempt_metrics = {}
            code_for_metrics = task_prompt + "\n" + output_code if output_code.strip() else ""
            if enable_metrics and bool(metrics_to_compute) and code_for_metrics.strip() and attempt_passed is True:
                attempt_metrics = analyze_code_metrics(code_for_metrics, metrics_to_compute=metrics_to_compute, code_for_loc=output_code)
           
            attempts.append({
                "attempt": i + 1,
                "output": output,
                "code": output_code,
                "passed": attempt_passed,
                "metrics": attempt_metrics,
            })
           
            # Display attempt
            with st.expander(f"Attempt {i+1} {'✅ PASSED' if attempt_passed else '❌ FAILED' if attempt_passed is False else '⚠️ ERROR'}", expanded=(i == 0)):
                st.code(output, language="python")
                st.markdown("**Extracted Code:**")
                st.code(output_code or "# (no code extracted)", language="python")
                if attempt_metrics:
                    st.markdown("**Metrics:**")
                    st.json(attempt_metrics)
           
            # Stop early when we get a pass (respecting max attempts)
            if attempt_passed is True:
                st.success(f"✅ Passed on attempt {i+1} — stopping early.")
                break
       
        progress_bar.progress(1.0)
       
        # Compute pass@k using the proper formula
        # n = actual attempts run (may be less than pass_at_k_value if we stopped early on pass)
        n = len(attempts)
        c = len(passed_attempts)  # Number of correct attempts
        k_eff = min(pass_at_k_value, n)  # Formula requires k <= n
        
        # Pass@1: just the first attempt
        pass_1 = 1 if attempts[0]["passed"] else (0 if attempts[0]["passed"] is False else None)
        
        # Pass@k using the estimate_pass_at_k function
        # Handle case where all attempts have None (evaluation error)
        all_none = all(att["passed"] is None for att in attempts)
        if all_none:
            pass_at_k = None
        else:
            pass_at_k = estimate_pass_at_k(n=n, c=c, k=k_eff)
       
        # Display summary
        st.markdown("## 📊 Pass@k Results")
        st.info(f"Using Pass@k formula: Pass@k = 1 - C(n-c, k) / C(n, k) where n={n}, c={c}, k={k_eff} (max budget: {pass_at_k_value})")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Pass@1", f"{pass_1}" if pass_1 is not None else "N/A")
        with col2:
            if pass_at_k is None:
                st.metric(f"Pass@{pass_at_k_value}", "N/A")
            else:
                st.metric(f"Pass@{pass_at_k_value}", f"{pass_at_k:.4f}")
        with col3:
            st.metric("Passed Attempts (c)", f"{c}/{n}")
        with col4:
            st.metric("Total Attempts (n)", f"{n}")
       
        if passed_attempts:
            st.success(f"✅ Passed attempts: {', '.join(map(str, passed_attempts))}")
        else:
            st.error("❌ No attempts passed")
       
        # Metrics summary table for all attempts
        if enable_metrics:
            st.markdown("## 📊 Metrics Summary (All Attempts)")
            metrics_data = []
            for attempt in attempts:
                m = attempt["metrics"]
                metrics_data.append({
                    "Attempt": attempt["attempt"],
                    "Passed": "✅" if attempt["passed"] else ("❌" if attempt["passed"] is False else "⚠️"),
                    "LOC": m.get("lines_of_code", "N/A"),
                    "Comment count": m.get("comment_count", "N/A"),
                    "CC": m.get("cyclomatic_complexity", "N/A"),
                    "Cognitive": m.get("cognitive_complexity", "N/A"),
                    "Halstead": m.get("halstead_effort", "N/A"),
                })
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)

        # Same outputs as single run: Full Model Output, Generated vs Canonical, Code Metrics
        st.markdown("## 🧠 Full Model Output")
        st.caption("From Attempt 1 (same layout as single run)")
        st.code(attempts[0]["output"], language="python")

        col_gen, col_can = st.columns(2)
        with col_gen:
            st.markdown("### 🧩 Generated Code (Used for Analysis)")
            st.code(attempts[0]["code"] or "# (no code extracted)", language="python")
        with col_can:
            st.markdown("### 📌 Canonical Solution")
            canonical = task.get("canonical_solution")
            if canonical:
                st.code(canonical, language="python")
            else:
                st.info("No canonical solution available for this task.")
        st.markdown("### 📜 Task Prompt")
        st.code(task_prompt, language="python")

        m = attempts[0].get("metrics") or {}
        if m and not all(v is None for v in m.values()):
            st.markdown("## 📊 Code Analysis Metrics")
            st.caption("From Attempt 1")
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.metric("Lines of Code", "N/A" if m.get("lines_of_code") is None else int(m["lines_of_code"]))
            with c2:
                st.metric("Comment count", "N/A" if m.get("comment_count") is None else int(m["comment_count"]))
            with c3:
                st.metric("Cyclomatic Complexity (CC)", "N/A" if m.get("cyclomatic_complexity") is None else round(float(m["cyclomatic_complexity"]), 4))
            with c4:
                st.metric("Cognitive Complexity", "N/A" if m.get("cognitive_complexity") is None else round(float(m["cognitive_complexity"]), 4))
            with c5:
                st.metric("Halstead Effort", "N/A" if m.get("halstead_effort") is None else round(float(m["halstead_effort"]), 4))
            st.json({k: m.get(k) for k in ["lines_of_code", "comment_count", "cyclomatic_complexity", "cognitive_complexity", "halstead_effort"]})
            df_metrics = pd.DataFrame([{
                "lines_of_code": m.get("lines_of_code"),
                "comment_count": m.get("comment_count"),
                "cyclomatic_complexity": m.get("cyclomatic_complexity"),
                "cognitive_complexity": m.get("cognitive_complexity"),
                "halstead_effort": m.get("halstead_effort"),
            }])
            st.download_button("Download Metrics CSV (Attempt 1)", df_metrics.to_csv(index=False), "metrics_attempt1.csv", "text/csv", key="dl_metrics_passatk")
        elif enable_metrics:
            st.markdown("## 📊 Code Analysis Metrics")
            st.info("No metrics computed for first attempt (missing deps, malformed code, or attempt did not pass).")

        # Compute similarity for this run (so we can save it to Supabase)
        similarity_score = None
        if enable_similarity and enable_rag and reference_text.strip() and attempts[0]["code"].strip():
            embedder = get_embedder()
            if embedder and util_mod:
                try:
                    emb_out = embedder.encode(attempts[0]["code"], convert_to_tensor=True)
                    emb_ref = embedder.encode(reference_text, convert_to_tensor=True)
                    similarity_score = util_mod.pytorch_cos_sim(emb_out, emb_ref).item()
                except Exception:
                    pass

        save_pass_at_k_run(
            model_name=selected_model,
            task_id=selected_task_id,
            task_prompt=task_prompt,
            canonical_solution=task.get("canonical_solution"),
            use_humaneval=use_humaneval,
            rag_enabled=enable_rag,
            pass_1=pass_1,
            pass_k=pass_at_k,
            k=pass_at_k_value,
            n_attempts=n,
            c_passed=c,
            attempts=attempts,
            raw_output_attempt1=attempts[0]["output"],
            extracted_code_attempt1=attempts[0]["code"],
            metrics_attempt1=attempts[0].get("metrics") or {},
            similarity_score=similarity_score,
        )

        # Use first attempt for similarity/metrics display
        output_code = attempts[0]["code"]
        code_for_metrics = task_prompt + "\n" + output_code if output_code.strip() else ""
        metrics = attempts[0]["metrics"]
       
    else:
        # Single run (original behavior)
        output = run_ollama(selected_model, combined_prompt)
        output_code = extract_python_code(output)
        if not output_code.strip() and output.strip():
            output_code = output.strip()


        st.markdown("## 🧠 Full Model Output")
        st.code(output, language="python")


        # For HumanEval runs, show generated code vs canonical solution side-by-side,
        # with the task prompt underneath. Otherwise, show just the extracted code.
        task_meta = None
        if use_humaneval and selected_task_id:
            try:
                problems_view = load_humaneval_problems() or {}
                task_meta = problems_view.get(selected_task_id)
            except Exception:
                traceback.print_exc()

        if task_meta:
            col_gen, col_can = st.columns(2)
            with col_gen:
                st.markdown("### 🧩 Generated Code (Used for Analysis)")
                st.code(output_code or "# (no code extracted)", language="python")
            with col_can:
                st.markdown("### 📌 Canonical Solution")
                canonical = task_meta.get("canonical_solution")
                if canonical:
                    st.code(canonical, language="python")
                else:
                    st.info("No canonical solution available for this task.")

            st.markdown("### 📜 Task Prompt")
            st.code(task_meta.get("prompt", ""), language="python")
        else:
            st.markdown("### 🧩 Extracted Code Used for Analysis")
            st.code(output_code or "# (no code extracted)", language="python")


        # IMPORTANT: HumanEval completions are often just function body (may start with 'return').
        # For metrics, we must analyze syntactically valid code:
        code_for_metrics = output_code
        if use_humaneval and selected_task_id:
            problems = load_humaneval_problems() or {}
            task = problems.get(selected_task_id)
            if task and task.get("prompt"):
                code_for_metrics = task["prompt"] + "\n" + output_code


        metrics = {}
        text_metrics = {}
        similarity_score = None
        pass_1 = None


        # ✅ Metrics
        if enable_metrics and bool(metrics_to_compute) and code_for_metrics.strip():
            st.markdown("## 📊 Code Analysis Metrics")
            m = analyze_code_metrics(code_for_metrics, metrics_to_compute=metrics_to_compute, code_for_loc=output_code)


            # If everything is None, show the same helpful message you saw before
            if all(v is None for v in m.values()):
                st.info("No metrics computed (missing deps or malformed code).")
            else:
                c1, c2, c3, c4, c5 = st.columns(5)
                with c1:
                    st.metric(
                        "Lines of Code",
                        "N/A" if m["lines_of_code"] is None else int(m["lines_of_code"]),
                    )
                with c2:
                    st.metric(
                        "Comment count",
                        "N/A" if m["comment_count"] is None else int(m["comment_count"]),
                    )
                with c3:
                    st.metric(
                        "Cyclomatic Complexity (CC)",
                        "N/A" if m["cyclomatic_complexity"] is None else round(float(m["cyclomatic_complexity"]), 4),
                    )
                with c4:
                    st.metric(
                        "Cognitive Complexity",
                        "N/A" if m["cognitive_complexity"] is None else round(float(m["cognitive_complexity"]), 4),
                    )
                with c5:
                    st.metric(
                        "Halstead Effort",
                        "N/A" if m["halstead_effort"] is None else round(float(m["halstead_effort"]), 4),
                    )


                st.json({k: m.get(k) for k in ["lines_of_code", "comment_count", "cyclomatic_complexity", "cognitive_complexity", "halstead_effort"]})


                df = pd.DataFrame([{
                    "lines_of_code": m.get("lines_of_code"),
                    "comment_count": m.get("comment_count"),
                    "cyclomatic_complexity": m.get("cyclomatic_complexity"),
                    "cognitive_complexity": m.get("cognitive_complexity"),
                    "halstead_effort": m.get("halstead_effort"),
                }])
                st.download_button("Download Metrics CSV", df.to_csv(index=False), "metrics.csv", "text/csv")


            metrics = m
        elif enable_metrics:
            st.warning("No code extracted — skipping code metrics.")


    # Similarity (RAG) - only for single run or first attempt
    if enable_similarity and enable_rag and reference_text.strip() and output_code.strip():
        st.markdown("## 🔍 Similarity vs Reference Document")
        embedder = get_embedder()
        if not embedder or not util_mod:
            st.warning("sentence-transformers not available — skipping semantic similarity.")
        else:
            emb_out = embedder.encode(output_code, convert_to_tensor=True)
            emb_ref = embedder.encode(reference_text, convert_to_tensor=True)
            similarity_score = util_mod.pytorch_cos_sim(emb_out, emb_ref).item()
            st.metric("Semantic similarity (0–1)", round(float(similarity_score), 4))

    # Persist simple single run separately
    if not enable_pass_at_k:
        task_prompt_for_log = None
        canonical_for_log = None
        if use_humaneval and selected_task_id:
            try:
                problems_for_log = load_humaneval_problems() or {}
                tmeta = problems_for_log.get(selected_task_id)
                if tmeta:
                    task_prompt_for_log = tmeta.get("prompt")
                    canonical_for_log = tmeta.get("canonical_solution")
            except Exception:
                # If this fails, we still log the rest
                traceback.print_exc()

        save_single_run_log(
            model_name=selected_model,
            prompt=combined_prompt,
            output=output,
            code=output_code,
            metrics=metrics,
            similarity_score=similarity_score,
            rag_enabled=enable_rag,
            use_humaneval=use_humaneval,
            selected_task_id=selected_task_id,
            task_prompt=task_prompt_for_log,
            canonical_solution=canonical_for_log,
        )


st.markdown("---")
st.markdown("## 🔄 Batch Automation")


batch_mode = st.checkbox("Enable Batch Mode", value=False)


if batch_mode:
    if not human_eval_data:
        st.error("HumanEval is required for batch mode. Please install human-eval package.")
    else:
        problems = load_humaneval_problems()
        if not problems:
            st.error("Failed to load HumanEval problems.")
        else:
            st.markdown("### 📋 Batch Configuration")
           
            # Default models
            default_models = ["codellama:7b", "mistral:7b", "qwen3:8b", "deepseek-r1:7b"]
            available_models, _ = list_ollama_models()
           
            # Combine available and default models for options (remove duplicates, preserve order)
            all_model_options = list(dict.fromkeys((available_models if available_models else []) + default_models))
            # Default to default_models that are in the options
            default_selected = [m for m in default_models if m in all_model_options]
           
            # Model selection
            st.markdown("**Select Models:**")
            selected_models = st.multiselect(
                "Choose models to evaluate",
                options=all_model_options if all_model_options else default_models,
                default=default_selected if default_selected else [],
                key="batch_models"
            )
           
            if not selected_models:
                st.warning("Please select at least one model.")
           
            # Default task IDs
            default_task_ids = ["HumanEval/7", "HumanEval/21", "HumanEval/33", "HumanEval/64", "HumanEval/81", "HumanEval/108", "HumanEval/163"]
            available_task_ids = sorted([tid for tid in problems.keys() if tid.startswith("HumanEval/")])
            # Filter default_task_ids to only include available ones
            default_task_ids = [tid for tid in default_task_ids if tid in available_task_ids]
           
            # Task selection
            st.markdown("**Select HumanEval Tasks:**")
            selected_task_ids = st.multiselect(
                "Choose HumanEval tasks",
                options=available_task_ids if available_task_ids else default_task_ids,
                default=default_task_ids if default_task_ids else [],
                key="batch_tasks"
            )
           
            if not selected_task_ids:
                st.warning("Please select at least one task.")
           
            # Batch options
            st.markdown("### ⚙️ Batch Options")
            attempts_per_task = st.number_input(
                "Number of attempts per (model, task)",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                key="batch_attempts_per_task",
            )
            batch_compute_metrics = st.checkbox("Compute metrics for each attempt", value=True, key="batch_metrics")

            batch_metrics_to_compute = set()
            if batch_compute_metrics:
                st.markdown("**Select which metrics to compute:**")
                metric_label_to_key = {
                    "Lines of Code (LOC)": "lines_of_code",
                    "Comment count": "comment_count",
                    "Cyclomatic Complexity (CC)": "cyclomatic_complexity",
                    "Cognitive Complexity": "cognitive_complexity",
                    "Halstead Effort": "halstead_effort",
                }
                default_metric_labels = [
                    "Lines of Code (LOC)",
                    "Comment count",
                    "Cyclomatic Complexity (CC)",
                    "Cognitive Complexity",
                    "Halstead Effort",
                ]
                selected_metric_labels = st.multiselect(
                    "Metrics",
                    options=list(metric_label_to_key.keys()),
                    default=default_metric_labels,
                    key="batch_metric_list",
                )
                batch_metrics_to_compute = {metric_label_to_key[label] for label in selected_metric_labels}

                if not batch_metrics_to_compute:
                    st.warning("No metrics selected — metrics will be skipped.")

            st.info("Batch pass@k uses k = number of attempts per (model, task).")
           
            # Run batch button
            if st.button("🚀 Run Batch Evaluation", type="primary", disabled=(not selected_models or not selected_task_ids)):
                if not selected_models or not selected_task_ids:
                    st.error("Please select at least one model and one task.")
                else:
                    total_jobs = len(selected_models) * len(selected_task_ids)
                    st.info(f"Starting batch: {len(selected_models)} models × {len(selected_task_ids)} tasks = {total_jobs} jobs ({attempts_per_task} attempt(s) each)")
                   
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                   
                    try:
                        # Determine whether we actually compute any metrics
                        effective_compute_metrics = batch_compute_metrics and bool(batch_metrics_to_compute)
                        rows, run_id, excel_path = run_batch_evaluation(
                            models=selected_models,
                            task_ids=selected_task_ids,
                            problems=problems,
                            compute_metrics=effective_compute_metrics,
                            metrics_to_compute=batch_metrics_to_compute if effective_compute_metrics else None,
                            attempts_per_task=int(attempts_per_task),
                            progress_bar=progress_bar,
                            status_text=status_text,
                        )
                       
                        progress_bar.progress(1.0)
                        status_text.text(f"✅ Batch completed! Run ID: {run_id}")
                       
                        st.success(f"Batch evaluation completed! Processed {len(rows)} jobs.")
                        st.info(f"Results saved to: {excel_path}")
                        st.info(f"Individual attempts logged to: experiment_logs.jsonl")
                        st.info("📐 Pass@k Formula: Pass@k = 1 - C(n-c, k) / C(n, k) where n=total attempts, c=correct attempts, k=samples")
                       
                        # Show summary
                        if rows:
                            summary_df = pd.DataFrame(rows)
                            st.markdown("### 📊 Batch Summary")
                            st.markdown("**Key columns:** `n_total_attempts` (n), `c_correct_attempts` (c), `k` (samples), `pass_k` (calculated using formula)")
                            st.dataframe(summary_df)
                           
                            # Download button (Excel format)
                            excel_buffer = BytesIO()
                            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                summary_df.to_excel(writer, sheet_name=run_id[:31], index=False)
                            excel_data = excel_buffer.getvalue()
                           
                            # Extract filename from path for download button
                            download_filename = os.path.basename(excel_path) if excel_path else f"batch_results_{run_id}.xlsx"
                           
                            st.download_button(
                                "Download Batch Results Excel",
                                excel_data,
                                download_filename,
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="download_batch"
                            )
                    except Exception as e:
                        st.error(f"Batch evaluation failed: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
