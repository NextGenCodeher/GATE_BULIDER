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
    page_title="Quantum Designer | Frost Edition",
    page_icon="❄️"
)

# --- Bluish-White Professional Theme CSS ---
st.markdown("""
    <style>
        /* Global Background - Cool Bluish White */
        .stApp {
            background-color: #F0F4F8;
            font-family: 'Inter', -apple-system, sans-serif;
        }
        
        /* Sidebar - Clean White with Blue Tint */
        section[data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #D1DBE5;
        }

        /* Headers */
        h1, h2, h3 {
            color: #1E3A5F !important;
            font-weight: 700 !important;
        }

        /* Metric Cards */
        div[data-testid="metric-container"] {
            background-color: #FFFFFF;
            border: 1px solid #E1E8F0;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02);
        }

        /* Buttons - Ice Blue Style */
        .stButton>button {
            border-radius: 8px;
            border: 1px solid #D1DBE5;
            background-color: #FFFFFF;
            color: #475569;
            transition: all 0.2s ease;
            font-weight: 500;
        }
        
        .stButton>button:hover {
            border-color: #3B82F6;
            color: #3B82F6;
            background-color: #EFF6FF;
            box-shadow: 0 0 10px rgba(59, 130, 246, 0.1);
        }

        /* Primary Action Button - Deep Corporate Blue */
        div[data-testid="stButton"] button[kind="primary"] {
            background-color: #2563EB !important;
            border: none !important;
            color: white !important;
            font-weight: 600 !important;
        }
        
        div[data-testid="stButton"] button[kind="primary"]:hover {
            background-color: #1D4ED8 !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        }

        /* Grid specific styling */
        .stProgress > div > div > div > div {
            background-color: #3B82F6;
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
    # Sphere colors updated to match bluish theme
    fig.add_trace(go.Surface(x=sphere_x, y=sphere_y, z=sphere_z,
                             colorscale=[[0, '#DBEAFE'], [1, '#DBEAFE']],
                             opacity=0.3, showscale=False))
    
    fig.add_trace(go.Scatter3d(x=[-1.2, 1.2], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='#94A3B8', width=2)))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[-1.2, 1.2], z=[0, 0], mode='lines', line=dict(color='#94A3B8', width=2)))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-1.2, 1.2], mode='lines', line=dict(color='#94A3B8', width=2)))
    
    fig.add_trace(go.Cone(x=[x], y=[y], z=[z], u=[x], v=[y], w=[z],
                          sizemode="absolute", sizeref=0.15, anchor="tip",
                          showscale=False, colorscale=[[0, '#2563EB'], [1, '#2563EB']]))
    
    fig.update_layout(
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='cube'),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# --- Sidebar ---
with st.sidebar:
    st.header('Designer Controls')
    num_qubits = st.slider('Qubits', 1, 5, 2)
    num_steps = st.slider('Depth', 5, 15, 10)
    num_shots = st.slider('Shots', 100, 4000, 1024)
    
    if 'circuit_grid' not in st.session_state or len(st.session_state.circuit_grid) != num_qubits:
        initialize_state(num_qubits, num_steps)

    if st.button('Reset Workspace', use_container_width=True):
        initialize_state(num_qubits, num_steps)
        st.rerun()

    st.markdown("### Gate Palette")
    st.write(f"Selected: **{st.session_state.active_gate}**")
    
    palette_gates = ['H', 'X', 'Y', 'Z', 'S', 'T', 'I', 'CNOT']
    cols = st.columns(2)
    for i, gate in enumerate(palette_gates):
        cols[i % 2].button(gate, on_click=set_active_gate, args=(gate,), use_container_width=True)

# --- Main Interface ---
st.title('⚛️ Quantum Circuit Simulator')
st.markdown("Select a gate from the sidebar, then click on the grid to place it")

# Grid Display
st.subheader("Quantum Circuit")
grid_container = st.container()
with grid_container:
    cols = st.columns([0.5] + [1] * num_steps)
    for t in range(num_steps):
        cols[t+1].markdown(f"<p style='text-align:center; color:#64748B;'>{t}</p>", unsafe_allow_html=True)
    
    for q in range(num_qubits):
        cols[0].markdown(f"**Q{q}**")
        for t in range(num_steps):
            gate = st.session_state.circuit_grid[q][t]
            cols[t+1].button(gate, key=f"g_{q}_{t}", on_click=place_gate, args=(q, t), use_container_width=True)

st.markdown("---")
if st.button('▶️ EXECUTE ANALYSIS', type="primary", use_container_width=True):
    try:
        with st.spinner("Processing Quantum State..."):
            # Circuit Construction
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

            # Results
            st.header("Results & Analysis")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Circuit Diagram")
                fig, ax = plt.subplots()
                qc.draw('mpl', ax=ax, style='iqx')
                st.pyplot(fig)
            
            with col2:
                st.subheader("Measurement Data")
                qc_m = qc.copy()
                qc_m.measure_all()
                counts = Aer.get_backend('qasm_simulator').run(qc_m, shots=num_shots).result().get_counts()
                
                st.metric("Most Probable State", max(counts, key=counts.get))
                
                hist = go.Figure(go.Bar(x=list(counts.keys()), y=list(counts.values()), marker_color='#3B82F6'))
                hist.update_layout(height=250, margin=dict(l=0,r=0,b=0,t=0), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(hist, use_container_width=True)

            # Bloch Spheres
            st.markdown("### Qubit State Analysis")
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
                    st.progress(np.real(rdm.data[1, 1]), text="Excitation")

    except Exception as e:
        st.error(f"Execution Error: {e}")

st.caption("Quantum Designer v2.1 | Blue-Frost Theme")