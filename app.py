import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import MultinomialHMM
import networkx as nx
import plotly.graph_objects as go
import warnings
import logging

# Suppress hmmlearn warnings
warnings.filterwarnings("ignore")
logging.getLogger("hmmlearn").setLevel(logging.CRITICAL)

st.set_page_config(page_title="HMM Baum-Welch Algorithm", layout="wide")

st.title("Hidden Markov Model - Baum-Welch Algorithm")
st.markdown("""
This web application implements a Hidden Markov Model (HMM) trained using the Baum–Welch (Expectation-Maximization) algorithm.
Enter your observation sequence and parameters below to train the model and visualize the results.
""")

# Sidebar for inputs
st.sidebar.header("Model Parameters")

obs_input = st.sidebar.text_input("Observation Sequence (comma-separated integers)", "0, 1, 1, 0, 1, 0, 1")
n_states = st.sidebar.number_input("Number of Hidden States", min_value=1, max_value=10, value=2, step=1)
max_iterations = st.sidebar.number_input("Number of Iterations", min_value=1, max_value=500, value=50, step=1)

if st.sidebar.button("Train Model"):
    try:
        # Parse observations
        observations = [int(x.strip()) for x in obs_input.split(",")]
        X = np.array(observations).reshape(-1, 1)
        
        # Create HMM model
        model = MultinomialHMM(
            n_components=n_states,
            n_iter=1,
            tol=1e-4,
            init_params="ste"
        )
        
        log_likelihoods = []
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(max_iterations):
            model.fit(X)
            score = model.score(X)
            log_likelihoods.append(score)
            
            # Update progress
            progress_bar.progress((i + 1) / max_iterations)
            status_text.text(f"Iteration {i + 1}/{max_iterations}")
            
        status_text.text("Training Complete!")
        
        # Display Results
        st.header("Training Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Initial Distribution (π)")
            st.dataframe(model.startprob_, width='stretch')
            
            st.subheader("Transition Matrix (A)")
            st.dataframe(model.transmat_, width='stretch')
            
        with col2:
            st.subheader("Emission Matrix (B)")
            st.dataframe(model.emissionprob_, width='stretch')
            
            st.subheader("Final Log Likelihood P(O | λ)")
            st.info(f"{model.score(X):.6f}")
            
        # Visualizations
        st.header("Visualizations")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Log Likelihood vs Iterations
        ax1.plot(range(1, max_iterations + 1), log_likelihoods, marker='o', linestyle='-')
        ax1.set_title("Log Likelihood P(O | λ) vs Iterations")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Log Likelihood")
        ax1.grid(True)
        
        # Plot 2: 1 - P(O | lambda) vs Iterations
        probabilities = np.exp(log_likelihoods)
        one_minus_probs = 1 - probabilities
        
        ax2.plot(range(1, max_iterations + 1), one_minus_probs, marker='x', linestyle='-', color='red')
        ax2.set_title("1 - P(O | λ) vs Iterations")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("1 - Probability")
        ax2.grid(True)
        
        st.pyplot(fig)
        
        # State Transition Diagram
        st.subheader("State Transition Diagram")
        
        # Use graphviz to generate a highly accurate and beautiful state transition diagram
        import graphviz
        
        # Create a directed graph
        dot = graphviz.Digraph(comment='State Transition Diagram')
        dot.attr(rankdir='LR', size='8,5')
        
        # Add nodes with styling
        dot.attr('node', shape='circle', style='filled', fillcolor='lightblue', 
                 color='black', fontname='Helvetica', fontsize='14', penwidth='2')
                 
        for i in range(n_states):
            dot.node(f"State {i}")
            
        # Add edges with styling
        dot.attr('edge', fontname='Helvetica', fontsize='12', color='gray40')
        
        for i in range(n_states):
            for j in range(n_states):
                prob = model.transmat_[i, j]
                if prob > 0.01: # Only show significant transitions
                    # Scale penwidth based on probability
                    penwidth = str(max(1.0, prob * 5.0))
                    dot.edge(f"State {i}", f"State {j}", label=f" {prob:.4f} ", penwidth=penwidth)
                    
        # Render the graph in Streamlit
        st.graphviz_chart(dot)
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
        import traceback
        st.error(traceback.format_exc())
        st.error("Please ensure your observation sequence contains only integers separated by commas.")
