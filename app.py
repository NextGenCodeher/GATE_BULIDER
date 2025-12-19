import streamlit as st
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import Aer
import plotly.graph_objects as go
from qiskit.quantum_info import DensityMatrix, partial_trace
import matplotlib.pyplot as plt

# --- 1. Page Configuration ---
st.set_page_config(layout="wide", page_title="Quantum Circuit Simulator")

# --- 2. Enhanced Interactive UI CSS (Bluish-White & Glassmorphism) ---
st.markdown("""
    <style>
        /* Main Background Gradient */
        .stApp {
            background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
        }
        
        /* Sidebar Styling: Dark Navy for Contrast */
        [data-testid="stSidebar"] {
            background-color: #102a43 !important;
        }
        [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
            color: #f0f4f8 !important;
        }

        /* Visibility Fix for Headings */
        h1 {
            color: #102a43 !important;
            font-family: 'Segoe UI', sans-serif;
            font-weight: 800 !important;
            padding-bottom: 20px;
        }
        h2, h3 {
            color: #243b53 !important;
            font-weight: 600 !important;
            margin-top: 30px;
        }

        /* Glassmorphism Buttons for the Circuit Grid */
        .stButton>button {
            background: rgba(255, 255, 255, 0.6);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.4);
            border-radius: 10px;
            color: #102a43;
            font-weight: 700;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        }
        
        /* Hover Animation */
        .stButton>button:hover {
            transform: translateY(-3px);
            background: #ffffff;
            border-color: #0d6efd;
            box-shadow: 0 10px 20px rgba(13, 110, 253, 0.15);
            color: #0d6efd;
        }

        /* Execute Button Styling */
        div.stButton > button:first-child[kind="primary"] {
            background: linear-gradient(45deg, #0d6efd, #00d4ff) !important;
            border: none !important;
            color: white !important;
            font-size: 1.1rem !important;
            height: 3.5rem !important;
            border-radius: 15px !important;
        }

        /* Results Card styling */
        .stMetric {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        }
    </style>
    """, unsafe_allow_html=True)

# --- Helper Functions & State Management ---

def initialize_state(num_qubits, num_steps):
    st.session_state.circuit_grid = [['I'] * num_steps for _ in range(num_qubits)]
    if 'active_gate' not in st.session_state:
        st.session_state.active_gate = 'H'

def set_active_gate(gate_symbol):
    st.session_state.active_gate = gate_symbol

def place_gate(q, t):
    active = st.session_state.active_gate
    if active == 'CNOT':
        st.session_state.circuit_grid[q][t] = '●'
        st.session_state.active_gate = '⊕'
    elif active == '⊕':
        st.session_state.circuit_grid[q][t] = '⊕'
        st.session_state.active_gate = 'H'
    else:
        st.session_state.circuit_grid[q][t] = active

def create_interactive_bloch_sphere(bloch_vector, title=""):
    x, y, z = bloch_vector
    fig = go.Figure()
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
    sphere_x = np.cos(u) * np.sin(v)
    sphere_y = np.sin(u) * np.sin(v)
    sphere_z = np.cos(v)
    fig.add_trace(go.Surface(x=sphere_x, y=sphere_y, z=sphere_z,
                             colorscale=[[0, '#c3dafe'], [1, '#c3dafe']],
                             opacity=0.2, showscale=False))
    fig.add_trace(go.Scatter3d(x=[-1.1, 1.1], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='#829ab1', width=2)))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[-1.1, 1.1], z=[0, 0], mode='lines', line=dict(color='#829ab1', width=2)))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-1.1, 1.1], mode='lines', line=dict(color='#829ab1', width=2)))
    fig.add_trace(go.Cone(x=[x], y=[y], z=[z], u=[x], v=[y], w=[z],
                          sizemode="absolute", sizeref=0.15, anchor="tip",
                          showscale=False, colorscale=[[0, '#ef4444'], [1, '#ef4444']]))
    fig.update_layout(
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='cube'),
        margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor='rgba(0,0,0,0)')
    return fig

# --- Header ---
st.title('⚛️ Quantum Circuit Simulator')
st.markdown("##### Select a gate from the sidebar and click the grid cells to build your circuit.")

# --- Sidebar ---
with st.sidebar:
    st.header('🛠️ Controls')
    num_qubits = st.slider('Number of Qubits', 1, 5, 2)
    num_steps = st.slider('Circuit Depth', 5, 15, 10)
    num_shots = st.number_input('Shots', 100, 8192, 1024)
    
    if 'circuit_grid' not in st.session_state or len(st.session_state.circuit_grid) != num_qubits:
        initialize_state(num_qubits, num_steps)

    if st.button('🔄 Reset Circuit', use_container_width=True):
        initialize_state(num_qubits, num_steps)
        st.rerun()

    st.header("✨ Gate Palette")
    st.write(f"Active Gate: **{st.session_state.active_gate}**")
    
    p_cols = st.columns(2)
    gates = ['H', 'X', 'Y', 'Z', 'S', 'T', 'I', 'CNOT']
    for i, g in enumerate(gates):
        p_cols[i % 2].button(g, key=f"pal_{g}", on_click=set_active_gate, args=(g,), use_container_width=True)

# --- Main Grid ---
st.header('🏗️ Circuit Builder')
grid_cols = st.columns(num_steps + 1)
grid_cols[0].write("Step:")

for i in range(num_steps):
    grid_cols[i + 1].markdown(f"**{i}**")

for q in range(num_qubits):
    grid_cols[0].markdown(f"**|q{q}⟩**")
    for t in range(num_steps):
        gate_label = st.session_state.circuit_grid[q][t]
        grid_cols[t + 1].button(gate_label, key=f"cell_{q}_{t}", on_click=place_gate, args=(q, t), use_container_width=True)

st.divider()

# --- Simulation Logic ---
if st.button('▶️ Execute Simulation', type="primary", use_container_width=True):
    try:
        with st.spinner("Processing Quantum Operations..."):
            qc = QuantumCircuit(num_qubits)
            for t in range(num_steps):
                c, tr = -1, -1
                for q in range(num_qubits):
                    if st.session_state.circuit_grid[q][t] == '●': c = q
                    if st.session_state.circuit_grid[q][t] == '⊕': tr = q
                
                if c != -1 and tr != -1: qc.cx(c, tr)
                elif c != -1 or tr != -1: raise ValueError(f"Incomplete CNOT at step {t}")
                else:
                    for q in range(num_qubits):
                        g = st.session_state.circuit_grid[q][t]
                        if g not in ['I', '●', '⊕']: getattr(qc, g.lower())(q)
            
            # --- Results Area ---
            c1, c2 = st.columns([1, 1])
            with c1:
                st.subheader("Circuit Diagram")
                fig_d, ax_d = plt.subplots()
                qc.draw('mpl', ax=ax_d, style='iqx')
                st.pyplot(fig_d)
            
            with c2:
                st.subheader("Measurement Statistics")
                qc_m = qc.copy()
                qc_m.measure_all()
                counts = Aer.get_backend('qasm_simulator').run(qc_m, shots=num_shots).result().get_counts()
                st.bar_chart(counts)

            st.subheader("Theoretical Bloch State")
            sv = Aer.get_backend('statevector_simulator').run(qc).result().get_statevector()
            dm = DensityMatrix(sv)
            
            b_cols = st.columns(num_qubits)
            for i in range(num_qubits):
                q_list = list(range(num_qubits))
                q_list.remove(i)
                rdm = partial_trace(dm, q_list)
                x = np.real(np.trace(rdm.data @ np.array([[0, 1], [1, 0]])))
                y = np.real(np.trace(rdm.data @ np.array([[0, -1j], [1j, 0]])))
                z = np.real(np.trace(rdm.data @ np.array([[1, 0], [0, -1]])))
                
                with b_cols[i]:
                    st.write(f"**Qubit {i}**")
                    st.plotly_chart(create_interactive_bloch_sphere([x, y, z]), use_container_width=True, key=f"b_{i}")

    except Exception as e:
        st.error(f"Execution Error: {e}")