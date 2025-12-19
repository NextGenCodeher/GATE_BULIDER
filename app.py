import streamlit as st
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import Aer
import plotly.graph_objects as go
from qiskit.quantum_info import DensityMatrix, partial_trace
import matplotlib.pyplot as plt

# --- 1. Page Configuration ---
st.set_page_config(layout="wide", page_title="Quantum Circuit Simulator")

# --- 2. Custom CSS for Bluish-White Background ---
st.markdown("""
    <style>
        /* Main background: Alice Blue / Bluish-White */
        .stApp {
            background-color: #f0f4f8;
        }
        
        /* Sidebar background: Soft Steel Blue */
        [data-testid="stSidebar"] {
            background-color: #e1e8f0;
        }

        /* Styling buttons to be more rounded and clean */
        .stButton>button {
            border-radius: 8px;
            transition: all 0.3s;
        }
        
        /* Header styling for better contrast */
        h1, h2, h3 {
            color: #1e3a5f;
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
                             colorscale=[[0, 'lightblue'], [1, 'lightblue']],
                             opacity=0.3, showscale=False))
    fig.add_trace(go.Scatter3d(x=[-1.2, 1.2], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='grey')))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[-1.2, 1.2], z=[0, 0], mode='lines', line=dict(color='grey')))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-1.2, 1.2], mode='lines', line=dict(color='grey')))
    fig.add_trace(go.Cone(x=[x], y=[y], z=[z], u=[x], v=[y], w=[z],
                          sizemode="absolute", sizeref=0.1, anchor="tip",
                          showscale=False, colorscale=[[0, 'red'], [1, 'red']]))
    fig.update_layout(
        title=dict(text=title, x=0.5), showlegend=False,
        scene=dict(xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, backgroundcolor="rgba(0,0,0,0)"),
                   yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, backgroundcolor="rgba(0,0,0,0)"),
                   zaxis=dict(showticklabels=False, showgrid=False, zeroline=False, backgroundcolor="rgba(0,0,0,0)"),
                   aspectmode='cube'),
        margin=dict(l=0, r=0, b=0, t=40))
    return fig

# --- Streamlit UI ---
st.title('⚛️ Quantum Circuit Simulator')
st.markdown("Select a gate from the sidebar, then click on the grid to place it.")

# --- Sidebar ---
with st.sidebar:
    st.header('Circuit Controls')
    num_qubits = st.slider('Number of Qubits', 1, 5, 2, key='num_qubits_slider')
    num_steps = st.slider('Circuit Depth', 5, 15, 10, key='num_steps_slider')
    num_shots = st.slider('Number of Shots', 100, 4000, 1024, key='shots_slider')
    
    if 'circuit_grid' not in st.session_state or len(st.session_state.circuit_grid) != num_qubits or len(st.session_state.circuit_grid[0]) != num_steps:
        initialize_state(num_qubits, num_steps)

    if st.button('Reset Circuit', use_container_width=True):
        initialize_state(num_qubits, num_steps)
        st.success("Circuit reset!")

    st.header("Gate Palette")
    st.write(f"Current Gate: **{st.session_state.active_gate}**")
    
    gate_palette_cols = st.columns(2)
    palette_gates = ['H', 'X', 'Y', 'Z', 'S', 'T', 'I', 'CNOT']
    for i, gate in enumerate(palette_gates):
        gate_palette_cols[i % 2].button(
            gate, on_click=set_active_gate, args=(gate,), use_container_width=True
        )
    if st.session_state.active_gate == '⊕':
        st.info("Click a grid cell to place the CNOT Target (⊕).")

# --- Main Circuit Grid UI ---
st.header('Quantum Circuit')
grid_cols = st.columns(num_steps + 1)
grid_cols[0].markdown("---") 

for i in range(num_steps):
    grid_cols[i + 1].markdown(f"<p style='text-align: center; font-weight: bold;'>{i}</p>", unsafe_allow_html=True)

for q in range(num_qubits):
    grid_cols[0].markdown(f"`|q{q}⟩`")
    for t in range(num_steps):
        gate_in_cell = st.session_state.circuit_grid[q][t]
        grid_cols[t + 1].button(
            gate_in_cell, key=f"cell_{q}_{t}", on_click=place_gate, args=(q, t), use_container_width=True
        )

# --- Execution Logic ---
if st.button('▶️ Execute Simulation', type="primary", use_container_width=True):
    try:
        with st.spinner("Simulating..."):
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
                    raise ValueError(f"Incomplete CNOT gate in time step {t}.")
                else:
                    for q in range(num_qubits):
                        gate = st.session_state.circuit_grid[q][t]
                        if gate != 'I' and gate != '●' and gate != '⊕':
                            getattr(qc, gate.lower())(q)
            
            st.success("✅ Simulation complete!")

            # --- Visualizations ---
            st.header("Circuit Diagram")
            fig_diag, ax_diag = plt.subplots()
            qc.draw('mpl', ax=ax_diag, style='iqx')
            st.pyplot(fig_diag)
            plt.close(fig_diag)
            
            # --- Measurement ---
            st.header("Measurement Outcomes")
            qc_measured = qc.copy()
            qc_measured.measure_all()
            qasm_backend = Aer.get_backend('qasm_simulator')
            counts = qasm_backend.run(qc_measured, shots=num_shots).result().get_counts()
            
            if counts:
                most_likely = max(counts, key=counts.get)
                st.metric(label="Most Probable Outcome", value=most_likely)
                
                sorted_counts = dict(sorted(counts.items()))
                hist_fig = go.Figure(go.Bar(x=list(sorted_counts.keys()), y=list(sorted_counts.values()), marker_color='#1e3a5f'))
                hist_fig.update_layout(title="Histogram of Results", xaxis_title="Classical State", yaxis_title="Counts", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(hist_fig, use_container_width=True)

            # --- Bloch Sphere Analysis ---
            st.header("Ideal State Analysis (Bloch Sphere)")
            sv_backend = Aer.get_backend('statevector_simulator')
            final_state = sv_backend.run(qc).result().get_statevector()
            final_dm = DensityMatrix(final_state)

            cols = st.columns(num_qubits)
            for i in range(num_qubits):
                q_list = list(range(num_qubits))
                q_list.remove(i)
                reduced_dm = partial_trace(final_dm, q_list)
                
                x = np.real(np.trace(reduced_dm.data @ np.array([[0, 1], [1, 0]])))
                y = np.real(np.trace(reduced_dm.data @ np.array([[0, -1j], [1j, 0]])))
                z = np.real(np.trace(reduced_dm.data @ np.array([[1, 0], [0, -1]])))
                
                with cols[i]:
                    st.subheader(f"Qubit {i}")
                    st.plotly_chart(create_interactive_bloch_sphere([x, y, z]), use_container_width=True, key=f"bloch_{i}")
                    st.text(f"|0⟩: {np.real(reduced_dm.data[0,0]):.3f}")
                    st.progress(float(np.real(reduced_dm.data[0,0])))
                    st.text(f"|1⟩: {np.real(reduced_dm.data[1,1]):.3f}")
                    st.progress(float(np.real(reduced_dm.data[1,1])))

    except Exception as e:
        st.error(f"Error: {e}")