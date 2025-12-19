import streamlit as st
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import Aer
import plotly.graph_objects as go
from qiskit.quantum_info import DensityMatrix, partial_trace
import matplotlib.pyplot as plt

# --- 1. Page Configuration ---
st.set_page_config(layout="wide", page_title="QuantumLab Ultra")

# --- 2. Ultra-Modern Professional CSS ---
st.markdown("""
    <style>
        /* Base Background: Soft Professional Grey-White */
        .stApp {
            background: #F0F2F5;
        }
        
        /* Side Navigation: Blue and Violet Gradient Mix */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #2D1B69 0%, #1A1A40 100%) !important;
            border-right: 1px solid rgba(255,255,255,0.1);
        }
        
        /* Sidebar Text Colors */
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
            color: #E0E0FF !important;
            font-family: 'Inter', sans-serif;
        }

        /* Main Workspace: Blurred White Glassmorphism */
        .main-glass-card {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.4);
            padding: 30px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
            margin-bottom: 25px;
        }

        /* Typography: Clean Black for Main Content */
        h1, h2, h3, p, label {
            color: #1A1A1A !important;
            font-family: 'Inter', sans-serif;
        }
        
        h1 { font-weight: 800 !important; letter-spacing: -1px; }

        /* Gate Palette Buttons: Sidebar Mix */
        .stButton > button {
            border-radius: 12px;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        /* Specifically Sidebar Palette Buttons */
        [data-testid="stSidebar"] .stButton > button {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: white !important;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background: linear-gradient(90deg, #6366F1, #A855F7);
            border: none;
            transform: scale(1.02);
        }

        /* Grid Buttons: Main Workspace */
        .main-glass-card .stButton > button {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            color: #1A1A1A;
        }
        .main-glass-card .stButton > button:hover {
            border-color: #6366F1;
            color: #6366F1;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.1);
        }

        /* Execute Button: Primary Action */
        div.stButton > button:first-child[kind="primary"] {
            background: linear-gradient(90deg, #4F46E5, #9333EA) !important;
            border: none !important;
            color: white !important;
            height: 3.5rem;
            width: 100%;
            font-size: 1.1rem !important;
            box-shadow: 0 10px 20px -5px rgba(79, 70, 229, 0.4) !important;
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
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
    fig.add_trace(go.Surface(x=np.cos(u)*np.sin(v), y=np.sin(u)*np.sin(v), z=np.cos(v),
                             colorscale=[[0, '#F1F5F9'], [1, '#F1F5F9']], opacity=0.2, showscale=False))
    fig.add_trace(go.Cone(x=[x], y=[y], z=[z], u=[x], v=[y], w=[z], sizemode="absolute", sizeref=0.2, 
                          anchor="tip", colorscale=[[0, '#6366F1'], [1, '#A855F7']], showscale=False))
    fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
                      margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor='rgba(0,0,0,0)', height=280)
    return fig

# --- Side Navigation ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/quantum-computing.png", width=80)
    st.title("Quantum Controls")
    
    num_qubits = st.slider('Number of Qubits', 1, 5, 2)
    num_steps = st.slider('Circuit Depth', 5, 15, 8)
    
    if 'circuit_grid' not in st.session_state or len(st.session_state.circuit_grid) != num_qubits:
        initialize_state(num_qubits, num_steps)

    st.markdown("### Gate Palette")
    st.caption(f"Current Tool: {st.session_state.active_gate}")
    p_cols = st.columns(2)
    gates = ['H', 'X', 'Y', 'Z', 'S', 'T', 'CNOT', 'I']
    for i, g in enumerate(gates):
        p_cols[i%2].button(g, key=f"p_{g}", on_click=set_active_gate, args=(g,), use_container_width=True)
    
    st.markdown("---")
    if st.button("Reset Designer"):
        initialize_state(num_qubits, num_steps)
        st.rerun()

# --- Main Workspace ---
st.title("Quantum Circuit Simulator")

# Circuit Designer Card
st.markdown('<div class="main-glass-card">', unsafe_allow_html=True)
st.subheader("Circuit Designer")
grid_cols = st.columns([0.6] + [1]*num_steps)

# Header Row for Steps
grid_cols[0].write("")
for i in range(num_steps):
    grid_cols[i+1].markdown(f"<center><small style='color:#64748B'>{i}</small></center>", unsafe_allow_html=True)

# Qubit Rows
for q in range(num_qubits):
    grid_cols[0].markdown(f"<p style='margin-top:10px; font-weight:700;'>|q{q}⟩</p>", unsafe_allow_html=True)
    for t in range(num_steps):
        label = st.session_state.circuit_grid[q][t]
        grid_cols[t+1].button(label, key=f"grid_{q}_{t}", on_click=place_gate, args=(q, t), use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)
if st.button('Execute Simulation', type="primary"):
    try:
        # Core Computation
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
        
        st.session_state.last_results = {
            'qc': qc,
            'counts': Aer.get_backend('qasm_simulator').run(qc.copy().measure_all(), shots=1024).result().get_counts(),
            'sv': Aer.get_backend('statevector_simulator').run(qc).result().get_statevector()
        }
    except Exception as e:
        st.error(f"Execution Error: {e}")

st.markdown('</div>', unsafe_allow_html=True)

# Results Section (Visible after execution)
if 'last_results' in st.session_state:
    res = st.session_state.last_results
    
    col_l, col_r = st.columns([1, 1.2])
    
    with col_l:
        st.markdown('<div class="main-glass-card">', unsafe_allow_html=True)
        st.subheader("Circuit Diagram")
        fig, ax = plt.subplots(figsize=(6, 4))
        res['qc'].draw('mpl', ax=ax, style='iqx')
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_r:
        st.markdown('<div class="main-glass-card">', unsafe_allow_html=True)
        st.subheader("Measurement Results")
        st.bar_chart(res['counts'], color="#6366F1")
        st.markdown('</div>', unsafe_allow_html=True)

    # Bloch Space Row
    st.subheader("Quantum Phase Space")
    dm = DensityMatrix(res['sv'])
    b_cols = st.columns(num_qubits)
    for i in range(num_qubits):
        with b_cols[i]:
            st.markdown('<div class="main-glass-card">', unsafe_allow_html=True)
            st.markdown(f"**Qubit {i}**")
            q_list = list(range(num_qubits)); q_list.remove(i)
            rdm = partial_trace(dm, q_list)
            x = np.real(np.trace(rdm.data @ np.array([[0, 1], [1, 0]])))
            y = np.real(np.trace(rdm.data @ np.array([[0, -1j], [1j, 0]])))
            z = np.real(np.trace(rdm.data @ np.array([[1, 0], [0, -1]])))
            st.plotly_chart(create_bloch_sphere([x, y, z]), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)