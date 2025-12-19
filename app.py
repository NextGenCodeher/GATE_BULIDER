import streamlit as st
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import Aer
import plotly.graph_objects as go
from qiskit.quantum_info import DensityMatrix, partial_trace
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    layout="wide", 
    page_title="Quantum Designer | Hybrid Edition",
    page_icon="⚛️"
)

# --- Hybrid Violet & Paper White CSS ---
st.markdown("""
    <style>
        /* Global Background - Clean Studio White */
        .stApp {
            background-color: #F8F9FC;
            font-family: 'Inter', -apple-system, sans-serif;
        }
        
        /* Sidebar - Deep Violet Gradient */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #2D1B69 0%, #160B39 100%) !important;
            border-right: 2px solid #7C3AED;
            color: white;
        }
        
        /* Sidebar text/headers */
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] .stMarkdown p {
            color: #E9D5FF !important;
        }

        /* Main Page Headers - Deep Indigo */
        h1, h2, h3 {
            color: #1E1B4B !important;
            font-weight: 800 !important;
        }
        
        .custom-instruction {
            color: #4338CA;
            font-size: 1.1rem;
            font-weight: 500;
            margin-bottom: 1.5rem;
            display: block;
        }

        /* Workspace Grid Buttons - Clean White with Violet Hover */
        .stButton>button {
            border-radius: 6px;
            border: 1px solid #E2E8F0;
            background-color: #FFFFFF;
            color: #1E293B;
            transition: all 0.2s ease;
            font-weight: 600;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        
        .stButton>button:hover {
            border-color: #7C3AED;
            color: #7C3AED;
            background-color: #F5F3FF;
            transform: translateY(-1px);
        }

        /* Sidebar Buttons - Light Violet Style */
        section[data-testid="stSidebar"] .stButton>button {
            background-color: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(167, 139, 250, 0.3);
            color: #F5F3FF;
        }

        section[data-testid="stSidebar"] .stButton>button:hover {
            background-color: #7C3AED;
            color: white;
            border-color: #A78BFA;
        }

        /* Execution Button - Solid Violet */
        div[data-testid="stButton"] button[kind="primary"] {
            background-color: #4F46E5 !important;
            border: none !important;
            color: white !important;
            padding: 0.6rem 2rem !important;
            font-size: 16px !important;
        }

        /* Metrics & Cards */
        div[data-testid="metric-container"] {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
    </style>
    """, unsafe_allow_html=True)

# --- Helper Functions ---

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

def create_interactive_bloch_sphere(bloch_vector):
    x, y, z = bloch_vector
    fig = go.Figure()
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
    sphere_x = np.cos(u) * np.sin(v)
    sphere_y = np.sin(u) * np.sin(v)
    sphere_z = np.cos(v)
    fig.add_trace(go.Surface(x=sphere_x, y=sphere_y, z=sphere_z,
                             colorscale=[[0, '#F5F3FF'], [1, '#DDD6FE']],
                             opacity=0.4, showscale=False))
    
    fig.add_trace(go.Scatter3d(x=[-1.1, 1.1], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='#A5B4FC', width=2)))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[-1.1, 1.1], z=[0, 0], mode='lines', line=dict(color='#A5B4FC', width=2)))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-1.1, 1.1], mode='lines', line=dict(color='#A5B4FC', width=2)))
    
    fig.add_trace(go.Cone(x=[x], y=[y], z=[z], u=[x], v=[y], w=[z],
                          sizemode="absolute", sizeref=0.15, anchor="tip",
                          showscale=False, colorscale=[[0, '#4F46E5'], [1, '#7C3AED']]))
    
    fig.update_layout(
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='cube'),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# --- Sidebar ---
with st.sidebar:
    st.markdown("## ⚙️ Designer Settings")
    num_qubits = st.slider('Qubits', 1, 5, 2)
    num_steps = st.slider('Depth', 5, 15, 10)
    num_shots = st.slider('Shots', 100, 4000, 1024)
    
    if 'circuit_grid' not in st.session_state or len(st.session_state.circuit_grid) != num_qubits:
        initialize_state(num_qubits, num_steps)

    if st.button('Reset Grid', use_container_width=True):
        initialize_state(num_qubits, num_steps)
        st.rerun()

    st.markdown("### 🛠️ Gate Palette")
    st.caption(f"Active Gate: {st.session_state.active_gate}")
    palette_gates = ['H', 'X', 'Y', 'Z', 'S', 'T', 'I', 'CNOT']
    cols = st.columns(2)
    for i, gate in enumerate(palette_gates):
        cols[i % 2].button(gate, on_click=set_active_gate, args=(gate,), use_container_width=True)

# --- Main Interface ---
st.title('⚛️ Quantum Circuit Simulator')
st.markdown('<span class="custom-instruction">Select a gate from the sidebar, then click on the grid to place it.</span>', unsafe_allow_html=True)

# Grid Display
st.subheader("Circuit Construction Grid")
with st.container():
    cols = st.columns([0.6] + [1] * num_steps)
    for t in range(num_steps):
        cols[t+1].markdown(f"<p style='text-align:center; color:#6366F1; font-weight:bold;'>{t}</p>", unsafe_allow_html=True)
    
    for q in range(num_qubits):
        cols[0].markdown(f"<p style='margin-top:8px; font-weight:700;'>Qubit {q}</p>", unsafe_allow_html=True)
        for t in range(num_steps):
            gate = st.session_state.circuit_grid[q][t]
            cols[t+1].button(gate, key=f"g_{q}_{t}", on_click=place_gate, args=(q, t), use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)
if st.button('▶️ EXECUTE SIMULATION', type="primary", use_container_width=True):
    try:
        with st.spinner("Processing Quantum State..."):
            qc = QuantumCircuit(num_qubits)
            for t in range(num_steps):
                ctrl, targ = -1, -1
                for q in range(num_qubits):
                    if st.session_state.circuit_grid[q][t] == '●': ctrl = q
                    if st.session_state.circuit_grid[q][t] == '⊕': targ = q
                if ctrl != -1 and targ != -1:
                    qc.cx(ctrl, targ)
                else:
                    for q in range(num_qubits):
                        gate = st.session_state.circuit_grid[q][t]
                        if gate not in ['I', '●', '⊕']:
                            getattr(qc, gate.lower())(q)

            # Results Section
            st.markdown("---")
            st.header("Simulation Analysis")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("Visual Circuit")
                fig, ax = plt.subplots(facecolor='#F8F9FC')
                qc.draw('mpl', ax=ax, style='iqx')
                st.pyplot(fig)
            
            with col2:
                st.subheader("Measurement Statistics")
                qc_m = qc.copy()
                qc_m.measure_all()
                counts = Aer.get_backend('qasm_simulator').run(qc_m, shots=num_shots).result().get_counts()
                st.metric("Peak Probability State", max(counts, key=counts.get))
                hist = go.Figure(go.Bar(x=list(counts.keys()), y=list(counts.values()), marker_color='#6366F1'))
                hist.update_layout(height=250, margin=dict(l=0,r=0,b=0,t=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(hist, use_container_width=True)

            # Bloch Spheres
            st.markdown("### Qubit State Visualization")
            sv = Aer.get_backend('statevector_simulator').run(qc).result().get_statevector()
            dm = DensityMatrix(sv)
            bloch_cols = st.columns(num_qubits)
            for i in range(num_qubits):
                q_list = list(range(num_qubits))
                q_list.remove(i)
                rdm = partial_trace(dm, q_list)
                x = np.real(np.trace(rdm.data @ np.array([[0, 1], [1, 0]])))
                y = np.real(np.trace(rdm.data @ np.array([[0, -1j], [1j, 0]])))
                z = np.real(np.trace(rdm.data @ np.array([[1, 0], [0, -1]])))
                with bloch_cols[i]:
                    st.markdown(f"<p style='text-align:center;'><b>Qubit {i}</b></p>", unsafe_allow_html=True)
                    st.plotly_chart(create_interactive_bloch_sphere([x, y, z]), use_container_width=True, key=f"b_{i}")
                    st.progress(float(np.real(rdm.data[1, 1])))

    except Exception as e:
        st.error(f"Error during execution: {e}")

st.caption("Quantum Designer v3.1 | Hybrid Studio Edition")