import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import warnings

# --- Page Config ---
st.set_page_config(page_title="📊 Extrema Finder", layout="centered")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Sidebar Help ---
with st.sidebar:
    st.title("📚 How to Use")
    st.markdown("""
**📌 Function Input Format:**
- Use `x` as the variable
- Use `**` for powers (e.g., `x**2`, not `x^2`)
- Use `sqrt(x)`, `log(x)`, `sin(x)`, `cos(x)`, etc.
- **Example:** `3*x**4 - 16*x**3 + 18*x**2`

**📌 Domain:**
- Enter any real-number bounds (e.g., -5 to 5)

💡 Click **Generate Plot** to visualize extrema!
""")

# --- Title & Inputs ---
st.title("🎯 Local and Global Extrema Finder")
st.markdown("Explore **local minima**, **local maxima**, and **global extrema** of any function.")

st.subheader("📝 Function & Domain Input")
user_func = st.text_input("Enter a function f(x):", "3*x**4 - 16*x**3 + 18*x**2")
col1, col2 = st.columns(2)
with col1:
    a = st.number_input("Lower Bound (a):", value=-1.0)
with col2:
    b = st.number_input("Upper Bound (b):", value=4.0)

# --- Generate Button ---
if st.button("🔍 Generate Plot"):
    x = sp.Symbol('x')
    try:
        f_expr = sp.sympify(user_func)
        f_prime = sp.diff(f_expr, x)
        f_double_prime = sp.diff(f_prime, x)
        f = sp.lambdify(x, f_expr, modules='numpy')

        # --- Critical Points ---
        crit_pts_raw = sp.solve(f_prime, x)
        crit_pts = []
        for pt in crit_pts_raw:
            try:
                val = float(pt)
                if np.isfinite(val) and a <= val <= b:
                    crit_pts.append(val)
            except:
                continue

        # --- Safe Evaluation ---
        def safe_eval(f, xs):
            result = []
            for x_val in xs:
                try:
                    y = f(x_val)
                    if np.isfinite(y):
                        result.append((x_val, y))
                except:
                    continue
            return result

        # --- Evaluate Function Safely ---
        x_vals = np.linspace(a, b, 500)
        safe_points = safe_eval(f, x_vals)
        if not safe_points:
            st.error("❌ Function could not be evaluated safely over the domain.")
            st.stop()

        x_safe, y_safe = zip(*safe_points)

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_safe, y_safe, label="f(x)", color='black')

        # --- Local Extrema via 2nd Derivative
        local_maxima, local_minima = [], []
        for pt in crit_pts:
            try:
                y_val = f(pt)
                second = f_double_prime.subs(x, pt)
                if np.isfinite(y_val) and second.is_real:
                    if second < 0:
                        local_maxima.append((pt, y_val))
                    elif second > 0:
                        local_minima.append((pt, y_val))
            except:
                continue

        for pt, y in local_maxima:
            ax.plot(pt, y, 'ro', label='Local Max' if 'Local Max' not in ax.get_legend_handles_labels()[1] else "")
        for pt, y in local_minima:
            ax.plot(pt, y, 'go', label='Local Min' if 'Local Min' not in ax.get_legend_handles_labels()[1] else "")

        # --- Global Extrema from Safe Points
        global_min = min(safe_points, key=lambda x: x[1])
        global_max = max(safe_points, key=lambda x: x[1])
        ax.plot(global_min[0], global_min[1], 'mx', markersize=10, label="Global Min")
        ax.plot(global_max[0], global_max[1], 'yX', markersize=10, label="Global Max")

        ax.set_title("🔍 Function Plot with Extrema", fontsize=14)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        # --- Function in LaTeX ---
        st.subheader("🧾 Function You Entered")
        st.latex(f"f(x) = {sp.latex(f_expr)}")

        # --- Summary of Extrema ---
        st.subheader("📌 Extrema Summary")

        if local_maxima:
            st.markdown("**Local Maxima:**")
            for x_val, y_val in local_maxima:
                st.write(f"f({x_val:.4f}) = {y_val:.4f}")
        if local_minima:
            st.markdown("**Local Minima:**")
            for x_val, y_val in local_minima:
                st.write(f"f({x_val:.4f}) = {y_val:.4f}")

        st.markdown("**Global Minimum:**")
        st.write(f"f({global_min[0]:.4f}) = {global_min[1]:.4f}")
        st.markdown("**Global Maximum:**")
        st.write(f"f({global_max[0]:.4f}) = {global_max[1]:.4f}")

        if not (local_maxima or local_minima):
            st.info("ℹ️ No local extrema detected within this domain.")

    except Exception as e:
        st.error(f"❌ Error while processing your input: {e}")

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; font-size: 0.9em;'>🧑‍🏫 Developed by Dr. Jishan Ahmed</div>", unsafe_allow_html=True)
