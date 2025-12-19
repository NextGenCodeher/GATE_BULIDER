import streamlit as st
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import Aer
import plotly.graph_objects as go
from qiskit.quantum_info import DensityMatrix, partial_trace
import matplotlib.pyplot as plt

# --- 1. Page Configuration ---
st.set_page_config(layout="wide", page_title="QuantumLab Pro")

# --- 2. Top-Tier Professional CSS ---
st.markdown("""
    <style>
        /* Base Background */
        .stApp {
            background-color: #F8FAFC;
        }
        
        /* Sidebar: High-End Professional Navy */
        [data-testid="stSidebar"] {
            background-color: #0F172A !important;
            border-right: 1px solid #E2E8F0;
        }
        [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
            color: #F8FAFC !important;
        }

        /* Professional Typography */
        h1 {
            color: #1E293B !important;
            font-family: 'Inter', sans-serif;
            font-weight: 800 !important;
            letter-spacing: -1px;
        }
        h2, h3 {
            color: #334155 !important;
            font-weight: 700 !important;
            letter-spacing: -0.5px;
        }

        /* The "Magic" Card Style */
        .quantum-card {
            background-color: #FFFFFF;
            padding: 24px;
            border-radius: 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            border: 1px solid #F1F5F9;
            margin-bottom: 20px;
        }

        /* Grid Button Redesign: Minimalist White */
        .stButton>button {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            color: #475569;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            border-color: #3B82F6;
            color: #3B82F6;
            background-color: #EFF6FF;
            transform: translateY(-1px);
        }

        /* Execute Button: Electric Blue */
        div.stButton > button:first-child[kind="primary"] {
            background: linear-gradient(135deg, #2563EB 0%, #3B82F6 100%) !important;
            border: none !important;
            font-weight: 700 !important;
            border-radius: 12px !important;
            box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3) !important;
        }

        /* Fix Plotly background */
        .js-plotly-plot {
            background-color: transparent !important;
        }
    </style>
    """, unsafe_allow_html=True)

# --- State Management ---
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

def create_bloch_sphere(vector):
    x, y, z = vector
    fig = go.Figure()
    # Transparent Sphere
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
    fig.add_trace(go.Surface(x=np.cos(u)*np.sin(v), y=np.sin(u)*np.sin(v), z=np.cos(v),
                             colorscale=[[0, '#E2E8F0'], [1, '#E2E8F0']], opacity=0.15, showscale=False))
    # Vector
    fig.add_trace(go.Cone(x=[x], y=[y], z=[z], u=[x], v=[y], w=[z], sizemode="absolute", sizeref=0.2, anchor="tip", colorscale=[[0, '#3B82F6'], [1, '#3B82F6']], showscale=False))
    fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
                      margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor='rgba(0,0,0,0)', height=250)
    return fig

# --- UI Layout ---
st.title("QuantumLab Pro")
st.markdown("<p style='color: #64748B; font-size: 1.1rem;'>Scientific-grade quantum circuit designer and simulator.</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Configuration")
    num_qubits = st.slider('Qubits', 1, 5, 2)
    num_steps = st.slider('Steps', 5, 15, 8)
    
    if 'circuit_grid' not in st.session_state or len(st.session_state.circuit_grid) != num_qubits:
        initialize_state(num_qubits, num_steps)

    st.markdown("---")
    st.markdown("### Gate Palette")
    st.caption(f"Active Tool: {st.session_state.active_gate}")
    cols = st.columns(2)
    gates = ['H', 'X', 'Y', 'Z', 'S', 'T', 'CNOT', 'I']
    for i, g in enumerate(gates):
        cols[i%2].button(g, on_click=set_active_gate, args=(g,), use_container_width=True)

# --- Main Workspace ---
st.markdown('<div class="quantum-card">', unsafe_allow_html=True)
st.subheader("Circuit Designer")
grid_cols = st.columns([0.5] + [1]*num_steps)
grid_cols[0].write("")
for i in range(num_steps): grid_cols[i+1].markdown(f"<center><b style='color:#94A3B8'>{i}</b></center>", unsafe_allow_html=True)

for q in range(num_qubits):
    grid_cols[0].markdown(f"<p style='margin-top:10px;'><b>q{q}</b></p>", unsafe_allow_html=True)
    for t in range(num_steps):
        label = st.session_state.circuit_grid[q][t]
        grid_cols[t+1].button(label, key=f"c_{q}_{t}", on_click=place_gate, args=(q, t), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

if st.button('Execute Simulation', type="primary", use_container_width=True):
    try:
        # Quantum Logic
        qc = QuantumCircuit(num_qubits)
        for t in range(num_steps):
            c, tr = -1, -1
            for q in range(num_qubits):
                if st.session_state.circuit_grid[q][t] == '●': c = q
                if st.session_state.circuit_grid[q][t] == '⊕': tr = q
            if c != -1 and tr != -1: qc.cx(c, tr)
            else:
                for q in range(num_qubits):
                    g = st.session_state.circuit_grid[q][t]
                    if g not in ['I', '●', '⊕']: getattr(qc, g.lower())(q)
        
        # --- Analytics Row ---
        col_diag, col_stats = st.columns([1, 1.2])
        
        with col_diag:
            st.markdown('<div class="quantum-card">', unsafe_allow_html=True)
            st.subheader("Physical Mapping")
            fig, ax = plt.subplots(figsize=(5, 3))
            qc.draw('mpl', ax=ax, style='iqx')
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_stats:
            st.markdown('<div class="quantum-card">', unsafe_allow_html=True)
            st.subheader("Probability Distribution")
            qc_m = qc.copy(); qc_m.measure_all()
            counts = Aer.get_backend('qasm_simulator').run(qc_m, shots=1024).result().get_counts()
            st.bar_chart(counts, color="#3B82F6")
            st.markdown('</div>', unsafe_allow_html=True)

        # --- Bloch Sphere Row ---
        st.subheader("Phase Space Visualization")
        sv = Aer.get_backend('statevector_simulator').run(qc).result().get_statevector()
        dm = DensityMatrix(sv)
        b_cols = st.columns(num_qubits)
        for i in range(num_qubits):
            with b_cols[i]:
                st.markdown('<div class="quantum-card">', unsafe_allow_html=True)
                st.write(f"**Qubit {i}**")
                q_list = list(range(num_qubits)); q_list.remove(i)
                rdm = partial_trace(dm, q_list)
                x = np.real(np.trace(rdm.data @ np.array([[0, 1], [1, 0]])))
                y = np.real(np.trace(rdm.data @ np.array([[0, -1j], [1j, 0]])))
                z = np.real(np.trace(rdm.data @ np.array([[1, 0], [0, -1]])))
                st.plotly_chart(create_bloch_sphere([x, y, z]), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Hardware Error: {e}")