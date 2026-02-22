import streamlit as st
import numpy as np
from hmmlearn.hmm import MultinomialHMM
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
        
        # Calculate Final Log Likelihood P(O | λ)
        final_log_likelihood = model.score(X)
        
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
            st.info(f"{final_log_likelihood:.6f}")
            
            st.subheader("Final Probability P(O | λ)")
            st.info(f"{np.exp(final_log_likelihood):.6e}")
            
        # Visualizations
        st.header("Visualizations")
        
        iterations = list(range(1, max_iterations + 1))
        
        # Plot 1: Log Likelihood vs Iterations
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=iterations, 
            y=log_likelihoods, 
            mode='lines+markers', 
            name='Log Likelihood',
            hovertemplate='Iteration: %{x}<br>Log Likelihood: %{y:.6f}<extra></extra>'
        ))
        fig1.update_layout(
            title="Log Likelihood P(O | λ) vs Iterations",
            xaxis_title="Iteration",
            yaxis_title="Log Likelihood",
            template="plotly_white",
            hovermode="x unified",
            margin=dict(l=40, r=40, t=40, b=40),
            xaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True),
            yaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True)
        )
        
        # Plot 2: 1 - P(O | lambda) vs Iterations
        probabilities = np.exp(log_likelihoods)
        one_minus_probs = 1 - probabilities
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=iterations, 
            y=one_minus_probs, 
            mode='lines+markers', 
            name='1 - Probability', 
            line=dict(color='red'),
            hovertemplate='Iteration: %{x}<br>1 - Probability: %{y:.6e}<extra></extra>'
        ))
        fig2.update_layout(
            title="1 - P(O | λ) vs Iterations",
            xaxis_title="Iteration",
            yaxis_title="1 - Probability",
            template="plotly_white",
            hovermode="x unified",
            margin=dict(l=40, r=40, t=40, b=40),
            xaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True),
            yaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True)
        )
        
        col_fig1, col_fig2 = st.columns(2)
        with col_fig1:
            st.plotly_chart(fig1, use_container_width=True)
        with col_fig2:
            st.plotly_chart(fig2, use_container_width=True)
        
        # State Transition Diagram
        st.subheader("State Transition Diagram")
        
        from streamlit_echarts import st_echarts
        
        # Prepare nodes
        nodes = []
        for i in range(n_states):
            nodes.append({
                "name": f"State {i}",
                "symbolSize": 60,
                "itemStyle": {
                    "color": "#E1F5FE",
                    "borderColor": "#0288D1",
                    "borderWidth": 2
                },
                "label": {
                    "show": True,
                    "fontSize": 14,
                    "fontWeight": "bold",
                    "color": "#000"
                }
            })
            
        # Prepare edges
        links = []
        for i in range(n_states):
            for j in range(n_states):
                prob = model.transmat_[i, j]
                if prob > 0.01:
                    # Calculate curvature to prevent overlapping of mutual edges
                    curveness = 0.2 if i != j else 0.5
                    
                    links.append({
                        "source": f"State {i}",
                        "target": f"State {j}",
                        "value": float(prob),
                        "label": {
                            "show": True,
                            "formatter": f"{prob:.3f}",
                            "fontSize": 12,
                            "color": "#000",
                            "backgroundColor": "rgba(255,255,255,0.8)",
                            "padding": 2,
                            "borderRadius": 3
                        },
                        "lineStyle": {
                            "width": max(1.5, prob * 6),
                            "curveness": curveness,
                            "color": "#00796B" if i == j else "#546E7A"
                        }
                    })
                    
        # ECharts configuration
        option = {
            "title": {
                "text": "Interactive State Transitions",
                "left": "center",
                "textStyle": {
                    "fontSize": 16,
                    "fontWeight": "normal"
                }
            },
            "tooltip": {
                "trigger": "item",
                "formatter": "{b}: {c}"
            },
            "animationDurationUpdate": 1500,
            "animationEasingUpdate": "quinticInOut",
            "series": [
                {
                    "type": "graph",
                    "layout": "circular" if n_states > 2 else "force",
                    "force": {
                        "repulsion": 1000,
                        "edgeLength": 200
                    } if n_states <= 2 else None,
                    "symbolSize": 50,
                    "roam": True, # Allows dragging and zooming
                    "label": {
                        "show": True
                    },
                    "edgeSymbol": ["circle", "arrow"],
                    "edgeSymbolSize": [4, 12],
                    "edgeLabel": {
                        "show": True,
                        "fontSize": 12
                    },
                    "data": nodes,
                    "links": links,
                    "lineStyle": {
                        "opacity": 0.9,
                        "width": 2,
                        "curveness": 0.2
                    },
                    "emphasis": {
                        "focus": "adjacency",
                        "lineStyle": {
                            "width": 8
                        }
                    }
                }
            ]
        }
        
        # Add a nice border around the interactive diagram
        st.markdown("""
        <style>
        .echarts-container {
            border: 2px solid black;
            border-radius: 5px;
            padding: 10px;
            background-color: white;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="echarts-container">', unsafe_allow_html=True)
        st_echarts(options=option, height="500px")
        st.markdown('</div>', unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
        import traceback
        st.error(traceback.format_exc())
        st.error("Please ensure your observation sequence contains only integers separated by commas.")
