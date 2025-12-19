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
    page_title="Quantum Nebula Designer",
    page_icon="🔮"
)

# --- Deep Violet & Purple 3D Theme CSS ---
st.markdown("""
    <style>
        /* Main Background - Deep Nebula Gradient */
        .stApp {
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            color: #E0E0E0;
            font-family: 'Inter', sans-serif;
        }
        
        /* Sidebar - Glassmorphism Violet */
        section[data-testid="stSidebar"] {
            background-color: rgba(31, 27, 65, 0.95) !important;
            border-right: 1px solid #7f5af0;
        }

        /* Titles and Instruction Text */
        h1, h2, h3 {
            color: #bd93f9 !important;
            text-shadow: 0px 0px 10px rgba(189, 147, 249, 0.5);
            font-weight: 800 !important;
        }
        
        .custom-instruction {
            color: #bd93f9;
            font-size: 1.1rem;
            font-weight: 500;
            margin-bottom: 1.5rem;
            display: block;
        }

        /* 3D-effect Metric Cards */
        div[data-testid="metric-container"] {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(127, 90, 240, 0.3);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            backdrop-filter: blur(4px);
        }

        /* Buttons - Interactive Purple 3D Style */
        .stButton>button {
            border-radius: 10px;
            border: 1px solid #7f5af0;
            background-color: rgba(127, 90, 240, 0.1);
            color: #fffffe;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            font-weight: 600;
            box-shadow: 0 4px 0px #4b30a1;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            background-color: #7f5af0;
            color: white;
            box-shadow: 0 6px 15px rgba(127, 90, 240, 0.4);
        }

        .stButton>button:active {
            transform: translateY(2px);
            box-shadow: 0 0px 0px #4b30a1;
        }

        /* Primary Action Button - Neon Violet */
        div[data-testid="stButton"] button[kind="primary"] {
            background: linear-gradient(45deg, #7f5af0, #bd93f9) !important;
            border: none !important;
            color: white !important;
            box-shadow: 0 0 20px rgba(127, 90, 240, 0.6) !important;
        }

        /* Inputs and Sliders */
        .stSlider [data-baseweb="slider"] {
            margin-bottom: 25px;
        }

        /* Grid specific styling */
        .stProgress > div > div > div > div {
            background-color: #bd93f9;
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
    
    # Glowy Violet Sphere
    fig.add_trace(go.Surface(x=sphere_x, y=sphere_y, z=sphere_z,
                             colorscale=[[0, '#24243e'], [1, '#7f5af0']],
                             opacity=0.4, showscale=False))
    
    # Neon Axes
    axis_style = dict(color='#bd93f9', width=3)
    fig.add_trace(go.Scatter3d(x=[-1.2, 1.2], y=[0, 0], z=[0, 0], mode='lines', line=axis_style))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[-1.2, 1.2], z=[0, 0], mode='lines', line=axis_style))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-1.2, 1.2], mode='lines', line=axis_style))
    
    # 3D Pointer
    fig.add_trace(go.Cone(x=[x], y=[y], z=[z], u=[x], v=[y], w=[z],
                          sizemode="absolute", sizeref=0.2, anchor="tip",
                          showscale=False, colorscale=[[0, '#ff00ff'], [1, '#bd93f9']]))
    
    fig.update_layout(
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='cube'),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# --- Sidebar ---
with st.sidebar:
    st.header('Nebula Controls')
    num_qubits = st.slider('Qubits', 1, 5, 2)
    num_steps = st.slider('Depth', 5, 15, 10)
    num_shots = st.slider('Shots', 100, 4000, 1024)
    
    if 'circuit_grid' not in st.session_state or len(st.session_state.circuit_grid) != num_qubits:
        initialize_state(num_qubits, num_steps)

    if st.button('Clear Circuit', use_container_width=True):
        initialize_state(num_qubits, num_steps)
        st.rerun()

    st.markdown("### Gate Palette")
    palette_gates = ['H', 'X', 'Y', 'Z', 'S', 'T', 'I', 'CNOT']
    cols = st.columns(2)
    for i, gate in enumerate(palette_gates):
        cols[i % 2].button(gate, on_click=set_active_gate, args=(gate,), use_container_width=True)

# --- Main Interface ---
st.title('⚛️ Quantum Nebula Simulator')
st.markdown('<span class="custom-instruction">Select a gate from the sidebar, then click on the grid to place it.</span>', unsafe_allow_html=True)

# Grid Display
st.subheader("Interactive Workspace")
grid_container = st.container()
with grid_container:
    cols = st.columns([0.6] + [1] * num_steps)
    for t in range(num_steps):
        cols[t+1].markdown(f"<p style='text-align:center; color:#bd93f9; font-weight:bold;'>{t}</p>", unsafe_allow_html=True)
    
    for q in range(num_qubits):
        cols[0].markdown(f"<h3 style='margin:0; padding-top:5px;'>Q{q}</h3>", unsafe_allow_html=True)
        for t in range(num_steps):
            gate = st.session_state.circuit_grid[q][t]
            cols[t+1].button(gate, key=f"g_{q}_{t}", on_click=place_gate, args=(q, t), use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)
if st.button('🚀 RUN NEBULA ENGINE', type="primary", use_container_width=True):
    try:
        with st.spinner("Decoding quantum interference..."):
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
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("Circuit Schematic")
                fig, ax = plt.subplots(facecolor='#1f1b41')
                qc.draw('mpl', ax=ax, style={'backgroundcolor': '#1f1b41', 'textcolor': '#bd93f9', 'linecolor': '#7f5af0'})
                st.pyplot(fig)
            
            with col2:
                st.subheader("Simulation Results")
                qc_m = qc.copy()
                qc_m.measure_all()
                counts = Aer.get_backend('qasm_simulator').run(qc_m, shots=num_shots).result().get_counts()
                st.metric("Dominant State", max(counts, key=counts.get))
                hist = go.Figure(go.Bar(x=list(counts.keys()), y=list(counts.values()), marker=dict(color='#bd93f9', line=dict(color='#7f5af0', width=1))))
                hist.update_layout(height=250, margin=dict(l=0,r=0,b=0,t=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#bd93f9"))
                st.plotly_chart(hist, use_container_width=True)

            # Bloch Spheres (3D Model Feel)
            st.markdown("### 3D Quantum State Mapping")
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
                    st.markdown(f"<p style='text-align:center; color:#bd93f9;'><b>Qubit {i}</b></p>", unsafe_allow_html=True)
                    st.plotly_chart(create_interactive_bloch_sphere([x, y, z]), use_container_width=True, key=f"b_{i}")
                    st.progress(float(np.real(rdm.data[1, 1])))

    except Exception as e:
        st.error(f"Engine Failure: {e}")

st.markdown("---")
st.caption("Quantum Nebula v3.0 | Deep Space Design")