import streamlit as st
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import Aer
import plotly.graph_objects as go
from qiskit.quantum_info import DensityMatrix, partial_trace
import matplotlib.pyplot as plt

# --- 1. Page Configuration ---
st.set_page_config(layout="wide", page_title="QuantumLab Pro Ultra")

# --- 2. Professional Bluish-Light & High-Contrast CSS ---
st.markdown("""
    <style>
        /* Base Background: Premium Soft Light Blue Gradient */
        .stApp {
            background: linear-gradient(135deg, #E0F2FE 0%, #F8FAFC 100%);
        }
        
        /* Sidebar: Deep Professional Blue-Violet Gradient */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1E3A8A 0%, #1E1B4B 100%) !important;
            border-right: 1px solid rgba(0,0,0,0.05);
        }
        
        /* Sidebar Text: Soft White for dark background contrast */
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
            color: #F0F9FF !important;
        }

        /* Main Workspace: Frosted White Glassmorphism Cards */
        .main-glass-card {
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.6);
            padding: 30px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05);
            margin-bottom: 25px;
        }

        /* TYPOGRAPHY: STRICT BLACK FOR ALL MAIN CONTENT */
        h1, h2, h3, p, label, .stMarkdown, b, strong, span, .stMetric label {
            color: #000000 !important;
            font-family: 'Inter', -apple-system, sans-serif;
        }
        
        h1 { font-weight: 900 !important; letter-spacing: -1.5px; padding-bottom: 10px; }
        h2, h3 { font-weight: 700 !important; margin-top: 20px; }

        /* Sidebar Buttons: Custom Violet/Blue Styling */
        [data-testid="stSidebar"] .stButton > button {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: white !important;
            border-radius: 10px;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background: linear-gradient(90deg, #60A5FA, #A78BFA);
            border: none;
            transform: scale(1.02);
        }

        /* Grid Buttons: Main Workspace (Black Text) */
        .main-glass-card .stButton > button {
            background: #FFFFFF;
            border: 1px solid #CBD5E1;
            color: #000000 !important;
            font-weight: 700;
            border-radius: 10px;
        }
        .main-glass-card .stButton > button:hover {
            border-color: #2563EB;
            color: #2563EB !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.1);
        }

        /* Primary Execution Button */
        div.stButton > button:first-child[kind="primary"] {
            background: linear-gradient(90deg, #2563EB, #7C3AED) !important;
            border: none !important;
            color: white !important;
            font-weight: 800 !important;
            height: 3.5rem;
            width: 100%;
            font-size: 1.1rem !important;
            box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.4) !important;
            border-radius: 15px !important;
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
                             colorscale=[[0, '#E2E8F0'], [1, '#E2E8F0']], opacity=0.3, showscale=False))
    fig.add_trace(go.Cone(x=[x], y=[y], z=[z], u=[x], v=[y], w=[z], sizemode="absolute", sizeref=0.2, 
                          anchor="tip", colorscale=[[0, '#2563EB'], [1, '#7C3AED']], showscale=False))
    fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
                      margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor='rgba(0,0,0,0)', height=280)
    return fig

# --- Side Navigation ---
with st.sidebar:
    st.markdown("# ⚛️ Quantum Controls")
    num_qubits = st.slider('Number of Qubits', 1, 5, 2)
    num_steps = st.slider('Circuit Depth', 5, 15, 8)
    
    if 'circuit_grid' not in st.session_state or len(st.session_state.circuit_grid) != num_qubits:
        initialize_state(num_qubits, num_steps)

    st.markdown("### Gate Palette")
    st.caption(f"Current Gate Selection: {st.session_state.active_gate}")
    p_cols = st.columns(2)
    gates = ['H', 'X', 'Y', 'Z', 'S', 'T', 'CNOT', 'I']
    for i, g in enumerate(gates):
        p_cols[i%2].button(g, key=f"p_{g}", on_click=set_active_gate, args=(g,), use_container_width=True)
    
    st.markdown("---")
    if st.button("Reset Designer", use_container_width=True):
        initialize_state(num_qubits, num_steps)
        st.rerun()

# --- Main Workspace ---
st.title("Quantum Circuit Simulator")
st.markdown("##### Build and simulate advanced quantum circuits in real-time.")

# Circuit Designer Workspace
st.markdown('<div class="main-glass-card">', unsafe_allow_html=True)
st.subheader("Circuit Builder")
grid_cols = st.columns([0.6] + [1]*num_steps)

grid_cols[0].write("")
for i in range(num_steps):
    grid_cols[i+1].markdown(f"<center><b style='color:#64748B'>{i}</b></center>", unsafe_allow_html=True)

for q in range(num_qubits):
    grid_cols[0].markdown(f"<p style='margin-top:10px; font-weight:800;'>|q{q}⟩</p>", unsafe_allow_html=True)
    for t in range(num_steps):
        label = st.session_state.circuit_grid[q][t]
        grid_cols[t+1].button(label, key=f"grid_{q}_{t}", on_click=place_gate, args=(q, t), use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)
if st.button('Execute Simulation', type="primary"):
    try:
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
        st.error(f"Hardware Logic Error: {e}")

st.markdown('</div>', unsafe_allow_html=True)

# Results Section
if 'last_results' in st.session_state:
    res = st.session_state.last_results
    col_l, col_r = st.columns([1, 1])
    
    with col_l:
        st.markdown('<div class="main-glass-card">', unsafe_allow_html=True)
        st.subheader("Physical Mapping")
        fig, ax = plt.subplots(figsize=(6, 4))
        res['qc'].draw('mpl', ax=ax, style='iqx')
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_r:
        st.markdown('<div class="main-glass-card">', unsafe_allow_html=True)
        st.subheader("State Probabilities")
        st.bar_chart(res['counts'], color="#2563EB")
        st.markdown('</div>', unsafe_allow_html=True)

    # Bloch Sphere Row
    st.markdown('<h2 style="text-align:center;">Quantum Phase Space</h2>', unsafe_allow_html=True)
    dm = DensityMatrix(res['sv'])
    b_cols = st.columns(num_qubits)
    for i in range(num_qubits):
        with b_cols[i]:
            st.markdown('<div class="main-glass-card">', unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align:center;'>Qubit {i}</h3>", unsafe_allow_html=True)
            q_list = list(range(num_qubits)); q_list.remove(i)
            rdm = partial_trace(dm, q_list)
            x = np.real(np.trace(rdm.data @ np.array([[0, 1], [1, 0]])))
            y = np.real(np.trace(rdm.data @ np.array([[0, -1j], [1j, 0]])))
            z = np.real(np.trace(rdm.data @ np.array([[1, 0], [0, -1]])))
            st.plotly_chart(create_bloch_sphere([x, y, z]), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)