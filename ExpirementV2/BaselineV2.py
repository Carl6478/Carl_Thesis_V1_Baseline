import os
import sys
import re
import math
import pandas as pd
import streamlit as st
import ollama

# 1. WINDOWS COMPATIBILITY PATCH
# Must be at the very top to prevent 'signal' errors
if sys.platform == "win32":
    import signal
    if not hasattr(signal, 'setitimer'):
        signal.SIGALRM = 14
        signal.ITIMER_REAL = 0
        def setitimer(which, seconds, interval=0.0):
            pass
        signal.setitimer = setitimer

# Set environment variable for execution
os.environ["ALLOW_EXECUTION"] = "1"

# 2. IMPORT REMAINING LIBRARIES
from human_eval.data import read_problems
from human_eval.execution import check_correctness
from radon.visitors import ComplexityVisitor
from radon.metrics import h_visit
import complexipy

# --- Helper Functions ---

def extract_code(text):
    """Extracts code from markdown backticks or returns raw text."""
    code_block = re.search(r"```(?:python)?\n(.*?)\n```", text, re.DOTALL)
    if code_block:
        return code_block.group(1).strip()
    return text.strip()

def calculate_pass_at_k(n, c, k):
    """Standard unbiased estimator for pass@k."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

def get_static_metrics(code):
    """Computes CC, LOC, Halstead, and Cognitive Complexity."""
    metrics = {}
    try:
        # Cyclomatic Complexity & LOC
        v = ComplexityVisitor.from_code(code)
        metrics['CC'] = sum(f.complexity for f in v.functions) if v.functions else 0
        metrics['LOC'] = len(code.splitlines())
        
        # Halstead Effort
        h = h_visit(code)
        metrics['Halstead'] = round(h.total.effort, 2)
        
        # Cognitive Complexity
        metrics['Cognitive'] = complexipy.get_cognitive_complexity(code)
    except Exception as e:
        return None
    return metrics

# --- Streamlit UI ---

st.set_page_config(page_title="LLM Benchmarker", layout="wide")
st.title("🚀 Ollama HumanEval Benchmarker")

with st.sidebar:
    st.header("Evaluation Settings")
    try:
        models_info = ollama.list()
        model_names = [m['name'] for m in models_info['models']]
        selected_model = st.selectbox("Select Ollama Model", model_names)
    except:
        st.error("Could not connect to Ollama. Ensure it is running.")
        selected_model = None

    k_val = st.number_input("Pass@k (k)", min_value=1, value=1)
    num_samples = st.number_input("Samples per problem (n)", min_value=k_val, value=2)
    num_problems = st.slider("Number of problems to test", 1, 164, 5)

if st.button("Run Benchmark") and selected_model:
    problems = read_problems()
    task_ids = list(problems.keys())[:num_problems]
    
    all_samples_data = []
    summary_data = []

    progress_bar = st.progress(0)
    
    for idx, task_id in enumerate(task_ids):
        st.subheader(f"Analyzing {task_id}")
        problem = problems[task_id]
        correct_count = 0
        
        cols = st.columns(num_samples)
        
        for i in range(num_samples):
            # Generate from Ollama
            response = ollama.generate(model=selected_model, prompt=problem['prompt'])
            raw_output = response['response']
            clean_code = extract_code(raw_output)
            
            # 1. Functional Check (Pass@k)
            # human_eval expects the completion without the prompt
            res = check_correctness(problem, clean_code, timeout=3.0)
            is_passed = res['passed']
            if is_passed:
                correct_count += 1
            
            # 2. Static Analysis
            m = get_static_metrics(clean_code)
            if m:
                m['task_id'] = task_id
                m['sample'] = i + 1
                m['result'] = "✅ Pass" if is_passed else "❌ Fail"
                all_samples_data.append(m)
            
            with cols[i]:
                st.text(f"Sample {i+1}")
                st.caption(m['result'] if m else "Error")

        # Calculate Pass@k Score for this problem
        pass_k_score = calculate_pass_at_k(num_samples, correct_count, k_val)
        summary_data.append({
            "Task ID": task_id,
            "Pass@k": f"{pass_k_score:.2%}",
            "Avg Cognitive": sum(d['Cognitive'] for d in all_samples_data if d['task_id'] == task_id) / num_samples
        })
        
        progress_bar.progress((idx + 1) / len(task_ids))

    # --- Final Results Display ---
    st.divider()
    st.success("Benchmarking Complete!")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.write("### Problem Summary")
        st.table(pd.DataFrame(summary_data))
        
    with col_right:
        st.write("### All Sample Details")
        st.dataframe(pd.DataFrame(all_samples_data))

elif not selected_model:
    st.warning("Please select a model from the sidebar.")