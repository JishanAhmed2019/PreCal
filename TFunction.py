import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import random

# --- Page Setup ---
st.set_page_config(page_title="Modular Function Transformer", layout="wide")
st.title("📈 Truly Modular Function Transformer")

# --- Sidebar Instructions ---
with st.sidebar:
    st.header("📘 How to Use This App")
    st.markdown(r"""
This app visualizes transformations of a function \( f(x) \), applying only the transformations you select.

### You Can:
- Apply shifts, stretches, and reflections independently
- Combine multiple effects
- See how each change affects the graph

### Bonus:
Try the **"Guess the Transformation"** challenge to test your understanding!
""")

# --- Function Input ---
st.subheader("1️⃣ Enter the Base Function \( f(x) \)")
user_input = st.text_input("Function:", value="x**2")
x = sp.Symbol('x')

try:
    base_expr = sp.sympify(user_input)
except Exception as e:
    st.error(f"Invalid function: {e}")
    st.stop()

# --- Transformation Controls ---
st.subheader("2️⃣ Choose Transformations")

col1, col2 = st.columns(2)
with col1:
    apply_hshift = st.checkbox("Horizontal Shift (h)")
    h = st.slider("Shift h (right if h > 0)", -10.0, 10.0, 0.0, 0.5) if apply_hshift else 0.0

    apply_hstretch = st.checkbox("Horizontal Stretch (b)")
    b = st.slider("Stretch b (b > 1 = compress)", 0.1, 5.0, 1.0, 0.1) if apply_hstretch else 1.0

    apply_reflect_y = st.checkbox("Reflect over Y-axis")
with col2:
    apply_vshift = st.checkbox("Vertical Shift (k)")
    k = st.slider("Shift k (up if k > 0)", -10.0, 10.0, 0.0, 0.5) if apply_vshift else 0.0

    apply_vstretch = st.checkbox("Vertical Stretch (a)")
    a = st.slider("Stretch a (a > 1 = taller)", 0.1, 5.0, 1.0, 0.1) if apply_vstretch else 1.0

    apply_reflect_x = st.checkbox("Reflect over X-axis")

# --- Build Transformed Expression ---
input_expr = x
if apply_hshift:
    input_expr = input_expr - h
if apply_hstretch:
    input_expr = b * input_expr
if apply_reflect_y:
    input_expr = -input_expr

transformed_expr = base_expr.subs(x, input_expr)

if apply_vstretch:
    transformed_expr = sp.Mul(a, transformed_expr, evaluate=False)
if apply_reflect_x:
    transformed_expr = sp.Mul(-1, transformed_expr, evaluate=False)
if apply_vshift:
    transformed_expr = sp.Add(transformed_expr, k, evaluate=False)

# --- Display Final Function ---
st.subheader("📘 Final Transformed Function")
st.latex(r"y = " + sp.latex(transformed_expr))

# --- Plotting Section ---
st.subheader("📊 Graph of Base vs Transformed Function")
x_vals = np.linspace(-15, 15, 1000)

try:
    f_base = sp.lambdify(x, base_expr, modules=["numpy"])
    f_trans = sp.lambdify(x, transformed_expr, modules=["numpy"])
    y_base = np.real(f_base(x_vals))
    y_trans = np.real(f_trans(x_vals))
except Exception as e:
    st.error(f"Error evaluating function: {e}")
    st.stop()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x_vals, y_base, '--', label=fr"Base: $f(x) = {sp.latex(base_expr)}$")
ax.plot(x_vals, y_trans, label=fr"Transformed: $y = {sp.latex(transformed_expr)}$", color='red', linewidth=2)
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.set_title("Function Transformation Visualization")

# Auto y-limits using both curves
combined_y = np.concatenate([y_base, y_trans])
finite_y = combined_y[np.isfinite(combined_y)]

if finite_y.size > 0:
    ymin, ymax = np.min(finite_y), np.max(finite_y)
    padding = (ymax - ymin) * 0.1 if ymax != ymin else 1
    ax.set_ylim([ymin - padding, ymax + padding])
else:
    ax.set_ylim([-20, 20])

st.pyplot(fig)

# --- Challenge Section ---
st.subheader("🎯 Guess the Transformation Challenge")

if "challenge_generated" not in st.session_state:
    st.session_state.challenge_generated = False

if st.button("🎲 Generate Challenge"):
    st.session_state.challenge_generated = True
    st.session_state.true_h = random.choice([0, 2, -3])
    st.session_state.true_k = random.choice([0, 3, -2])
    st.session_state.true_a = random.choice([1, 2])
    st.session_state.true_b = random.choice([1, 0.5])
    st.session_state.reflect_x = random.choice([True, False])
    st.session_state.reflect_y = random.choice([True, False])

if st.session_state.challenge_generated:
    challenge_input = x
    if st.session_state.true_h != 0:
        challenge_input = challenge_input - st.session_state.true_h
    if st.session_state.true_b != 1:
        challenge_input = st.session_state.true_b * challenge_input
    if st.session_state.reflect_y:
        challenge_input = -challenge_input

    challenge_expr = base_expr.subs(x, challenge_input)
    if st.session_state.true_a != 1:
        challenge_expr = sp.Mul(st.session_state.true_a, challenge_expr, evaluate=False)
    if st.session_state.reflect_x:
        challenge_expr = sp.Mul(-1, challenge_expr, evaluate=False)
    if st.session_state.true_k != 0:
        challenge_expr = sp.Add(challenge_expr, st.session_state.true_k, evaluate=False)

    st.latex(r"\text{Transformed Function: } y = " + sp.latex(challenge_expr))
    st.info("Can you guess the values of h, k, a, b and whether it was reflected?")
