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
    page_title="Quantum Architect | Professional Edition",
    page_icon="⚛️"
)

# --- Professional UI Styling (CSS Injection) ---
st.markdown("""
    <style>
        /* Global Background & Typography */
        .stApp {
            background-color: #FBFBFB;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* Sidebar Refinement */
        section[data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #ECECEC;
        }
        
        /* Headers & Metrics */
        h1, h2, h3 {
            color: #1A1A1A !important;
            font-weight: 600 !important;
            letter-spacing: -0.5px;
        }
        
        div[data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            color: #0D6EFD !important;
        }

        /* Professional Button Styling */
        .stButton>button {
            border-radius: 4px;
            border: 1px solid #E0E0E0;
            background-color: #FFFFFF;
            color: #444444;
            font-weight: 500;
            transition: all 0.3s ease;
            height: 38px;
        }
        
        .stButton>button:hover {
            border-color: #0D6EFD;
            color: #0D6EFD;
            background-color: #F0F7FF;
        }

        /* Primary Execution Button (Deep Black/Professional) */
        div[data-testid="stButton"] button[kind="primary"] {
            background-color: #1A1A1A !important;
            border: none !important;
            color: #FFFFFF !important;
            padding: 0.7rem 1rem !important;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 14px;
        }
        
        div[data-testid="stButton"] button[kind="primary"]:hover {
            background-color: #444444 !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }

        /* Analysis Cards */
        .qubit-card {
            background-color: #FFFFFF;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #E9E9E9;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            margin-bottom: 20px;
        }

        /* Grid Borders */
        div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column"] {
            gap: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

# --- Gate Definitions ---
GATE_DEFINITIONS = {
    'I': {'name': 'Identity', 'color': '#6c757d'},
    'H': {'name': 'Hadamard', 'color': '#0d6efd'},
    'X': {'name': 'Pauli-X', 'color': '#dc3545'},
    'Y': {'name': 'Pauli-Y', 'color': '#dc3545'},
    'Z': {'name': 'Pauli-Z', 'color': '#dc3545'},
    'S': {'name': 'S Gate', 'color': '#ffc107'},
    'T': {'name': 'T Gate', 'color': '#ffc107'},
    '●': {'name': 'Control', 'color': '#198754'},
    '⊕': 'Target (X)',
}

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
                             colorscale=[[0, '#E5E9F0'], [1, '#E5E9F0']],
                             opacity=0.2, showscale=False))
    fig.add_trace(go.Scatter3d(x=[-1.1, 1.1], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='#D8DEE9', width=2)))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[-1.1, 1.1], z=[0, 0], mode='lines', line=dict(color='#D8DEE9', width=2)))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-1.1, 1.1], mode='lines', line=dict(color='#D8DEE9', width=2)))
    fig.add_trace(go.Cone(x=[x], y=[y], z=[z], u=[x], v=[y], w=[z],
                          sizemode="absolute", sizeref=0.15, anchor="tip",
                          showscale=False, colorscale=[[0, '#5E81AC'], [1, '#5E81AC']]))
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14, color='#4C566A')),
        showlegend=False,
        scene=dict(xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, backgroundcolor="rgba(0,0,0,0)"),
                   yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, backgroundcolor="rgba(0,0,0,0)"),
                   zaxis=dict(showticklabels=False, showgrid=False, zeroline=False, backgroundcolor="rgba(0,0,0,0)"),
                   aspectmode='cube'),
        margin=dict(l=0, r=0, b=0, t=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# --- Sidebar ---
with st.sidebar:
    st.title("Settings")
    st.markdown("---")
    num_qubits = st.slider('Qubits', 1, 5, 2, key='num_qubits_slider')
    num_steps = st.slider('Depth', 5, 15, 10, key='num_steps_slider')
    num_shots = st.slider('Sim Shots', 100, 4000, 1024, key='shots_slider')
    
    if 'circuit_grid' not in st.session_state or len(st.session_state.circuit_grid) != num_qubits or len(st.session_state.circuit_grid[0]) != num_steps:
        initialize_state(num_qubits, num_steps)

    if st.button('Clear Workspace', use_container_width=True):
        initialize_state(num_qubits, num_steps)
        st.rerun()

    st.markdown("### Gate Palette")
    st.caption(f"Active Selection: {st.session_state.active_gate}")
    
    gate_palette_cols = st.columns(2)
    palette_gates = ['H', 'X', 'Y', 'Z', 'S', 'T', 'I', 'CNOT']
    for i, gate in enumerate(palette_gates):
        gate_palette_cols[i % 2].button(
            gate, on_click=set_active_gate, args=(gate,), use_container_width=True
        )

# --- Main Circuit Grid UI ---
st.title('⚛️ Quantum Designer')
st.caption("Construct and analyze quantum circuits in a laboratory environment.")

st.markdown("### Circuit Blueprint")
grid_cols = st.columns([0.5] + [1] * num_steps)
grid_cols[0].write("") 

for i in range(num_steps):
    grid_cols[i + 1].markdown(f"<p style='text-align: center; color: #888; font-size: 12px;'>T{i}</p>", unsafe_allow_html=True)

for q in range(num_qubits):
    grid_cols[0].markdown(f"**Q{q}**")
    for t in range(num_steps):
        gate_in_cell = st.session_state.circuit_grid[q][t]
        grid_cols[t + 1].button(
            gate_in_cell, key=f"cell_{q}_{t}", on_click=place_gate, args=(q, t), use_container_width=True
        )

st.markdown("---")
execute_btn = st.button('▶️ RUN SIMULATION', type="primary", use_container_width=True)

# --- Execution Logic ---
if execute_btn:
    try:
        with st.spinner("Calculating quantum states..."):
            qc = QuantumCircuit(num_qubits)
            for t in range(num_steps):
                control_qubit = -1
                target_qubit = -1
                for q in range(num_qubits):
                    gate = st.session_state.circuit_grid[q][t]
                    if gate == '●': control_qubit = q
                    elif gate == '⊕': target_qubit = q
                
                if control_qubit != -1 and target_qubit != -1:
                    qc.cx(control_qubit, target_qubit)
                elif control_qubit != -1 or target_qubit != -1:
                    raise ValueError(f"Dangling CNOT at step {t}")
                else:
                    for q in range(num_qubits):
                        gate = st.session_state.circuit_grid[q][t]
                        if gate not in ['I', '●', '⊕']:
                            getattr(qc, gate.lower())(q)

            # --- Results Display ---
            col_res1, col_res2 = st.columns([1, 1])
            
            with col_res1:
                st.subheader("Circuit Schematic")
                fig, ax = plt.subplots(figsize=(5, 3))
                qc.draw('mpl', ax=ax, style='bw') # Black and white for pro look
                st.pyplot(fig)
                plt.close(fig)

            with col_res2:
                st.subheader("Statistical Outcome")
                qc_measured = qc.copy()
                qc_measured.measure_all()
                qasm_backend = Aer.get_backend('qasm_simulator')
                counts = qasm_backend.run(qc_measured, shots=num_shots).result().get_counts()
                
                most_likely = max(counts, key=counts.get)
                st.metric("Peak State", most_likely)
                
                sorted_counts = dict(sorted(counts.items()))
                hist_fig = go.Figure(go.Bar(
                    x=list(sorted_counts.keys()), 
                    y=list(sorted_counts.values()),
                    marker_color='#1A1A1A'
                ))
                hist_fig.update_layout(height=250, margin=dict(l=0,r=0,b=0,t=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(hist_fig, use_container_width=True)

            # --- Bloch Sphere Analysis ---
            st.markdown("### Theoretical Analysis (Pre-Measurement)")
            statevector_backend = Aer.get_backend('statevector_simulator')
            final_state = statevector_backend.run(qc).result().get_statevector()
            final_dm = DensityMatrix(final_state)

            cols = st.columns(num_qubits)
            for i in range(num_qubits):
                q_list = list(range(num_qubits))
                q_list.remove(i)
                reduced_dm = partial_trace(final_dm, q_list)
                
                x = np.real(np.trace(reduced_dm.data @ np.array([[0, 1], [1, 0]])))
                y = np.real(np.trace(reduced_dm.data @ np.array([[0, -1j], [1j, 0]])))
                z = np.real(np.trace(reduced_dm.data @ np.array([[1, 0], [0, -1]])))
                
                prob_1 = np.real(reduced_dm.data[1, 1])
                purity = np.real(np.trace(reduced_dm.data @ reduced_dm.data))

                with cols[i]:
                    st.markdown(f"**QUBIT {i}**")
                    st.plotly_chart(create_interactive_bloch_sphere([x, y, z]), use_container_width=True, key=f"sphere_{i}")
                    st.progress(prob_1, text=f"Excitation Probability (|1⟩): {prob_1:.2%}")
                    st.caption(f"Purity: {purity:.3f} | Vector: [{x:.1f}, {y:.1f}, {z:.1f}]")

    except Exception as e:
        st.error(f"Engine Error: {e}")

st.markdown("---")
st.caption("Quantum Designer v2.0 | Built for High-Performance Simulation")