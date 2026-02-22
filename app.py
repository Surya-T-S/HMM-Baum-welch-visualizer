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
        import math
        
        # Intelligently place nodes in a perfect circle
        radius = 100 
        center_x = 0
        center_y = 0
        
        for i in range(n_states):
            # Calculate angle for this node (evenly spaced around a circle)
            angle = (2 * math.pi * i) / n_states
            
            # Calculate x and y coordinates
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            
            nodes.append({
                "name": f"State {i}",
                "value": [x, y], # Coordinates for cartesian2d
                "tooltip": {"formatter": "{b}"}, # Only show name on hover
                "symbolSize": 80,
                "itemStyle": {
                    "color": "#E1F5FE",
                    "borderColor": "#0288D1",
                    "borderWidth": 3
                },
                "label": {
                    "show": True,
                    "fontSize": 16,
                    "fontWeight": "bold",
                    "color": "#000",
                    "position": "inside",
                    "formatter": "{b}" # Show name
                }
            })
            
        # Prepare edges and animation lines
        links = []
        lines_data = []
        
        for i in range(n_states):
            for j in range(n_states):
                prob = model.transmat_[i, j]
                if prob > 0.01:
                    # Calculate curvature to prevent overlapping of mutual edges
                    if i == j:
                        curveness = 0.6 # Self loop (tight curve)
                    else:
                        curveness = 0.25
                    
                    # 1. Static visible edge for the graph
                    links.append({
                        "source": f"State {i}",
                        "target": f"State {j}",
                        "value": float(prob),
                        "tooltip": {"formatter": "{b}: {c}"}, # Show probability on hover
                        "label": {
                            "show": True,
                            "formatter": f"{prob:.3f}",
                            "fontSize": 14,
                            "fontWeight": "bold",
                            "color": "#000",
                            "backgroundColor": "rgba(255,255,255,1.0)",
                            "padding": [4, 6],
                            "borderRadius": 4,
                            "borderWidth": 1,
                            "borderColor": "#000"
                        },
                        "lineStyle": {
                            "width": max(2.0, prob * 8),
                            "curveness": curveness,
                            "color": "#00796B" if i == j else "#546E7A",
                            "opacity": 0.8
                        }
                    })
                    
                    # 2. Animated particle edge (only for transitions between different states)
                    if i != j:
                        # Calculate exact start and end points on the edge of the circle (radius 40)
                        # instead of the center of the circle, so the animation stops at the border
                        dx = nodes[j]["value"][0] - nodes[i]["value"][0]
                        dy = nodes[j]["value"][1] - nodes[i]["value"][1]
                        dist = math.hypot(dx, dy)
                        
                        if dist > 0:
                            # Node radius is 40 (symbolSize is 80, so radius is 40)
                            # We add a tiny bit of padding (45) so it stops right at the border
                            node_radius = 45
                            
                            # Direction vector
                            dir_x = dx / dist
                            dir_y = dy / dist
                            
                            # Start point (edge of source node)
                            start_x = nodes[i]["value"][0] + dir_x * node_radius
                            start_y = nodes[i]["value"][1] + dir_y * node_radius
                            
                            # End point (edge of target node)
                            end_x = nodes[j]["value"][0] - dir_x * node_radius
                            end_y = nodes[j]["value"][1] - dir_y * node_radius
                            
                            lines_data.append({
                                "coords": [
                                    [start_x, start_y],
                                    [end_x, end_y]
                                ],
                                "lineStyle": {
                                    "curveness": curveness
                                },
                                "effect": {
                                    "color": "#546E7A" # Match the color of the line
                                }
                            })
                    
        # ECharts configuration
        option = {
            "title": {
                "text": "Interactive State Transitions (Drag background to pan, scroll to zoom)",
                "left": "center",
                "textStyle": {
                    "fontSize": 16,
                    "fontWeight": "bold",
                    "color": "#333"
                }
            },
            "tooltip": {
                "trigger": "item"
            },
            "xAxis": {
                "show": False,
                "type": "value",
                "min": -radius * 1.5,
                "max": radius * 1.5
            },
            "yAxis": {
                "show": False,
                "type": "value",
                "min": -radius * 1.5,
                "max": radius * 1.5
            },
            "dataZoom": [
                {
                    "type": "inside",
                    "xAxisIndex": 0,
                    "yAxisIndex": 0,
                    "zoomOnMouseWheel": True,
                    "moveOnMouseMove": True,
                    "moveOnMouseWheel": False
                }
            ],
            "animationDurationUpdate": 1500,
            "animationEasingUpdate": "quinticInOut",
            "series": [
                {
                    "type": "graph",
                    "coordinateSystem": "cartesian2d", # Use cartesian to allow panning/zooming and align with lines
                    "symbolSize": 80,
                    "edgeSymbol": ["none", "arrow"],
                    "edgeSymbolSize": [0, 16],
                    "data": nodes,
                    "links": links,
                    "emphasis": {
                        "focus": "adjacency",
                        "lineStyle": {
                            "width": 8,
                            "opacity": 1,
                            "shadowBlur": 10,
                            "shadowColor": "rgba(0, 0, 0, 0.5)"
                        },
                        "label": {
                            "fontSize": 18,
                            "fontWeight": "bold"
                        }
                    }
                },
                {
                    "type": "lines",
                    "coordinateSystem": "cartesian2d",
                    "zlevel": 2,
                    "effect": {
                        "show": False, # Hide animation by default
                        "period": 4, # Slower, more subtle animation
                        "trailLength": 0.3, # Longer, softer tail
                        "symbol": "circle", # Use a soft circle instead of a sharp arrow
                        "symbolSize": 4 # Smaller, more subtle particle
                    },
                    "lineStyle": {
                        "color": "transparent", # Hide the static line, only show the animated particle
                        "width": 0
                    },
                    "emphasis": {
                        "focus": "adjacency",
                        "effect": {
                            "show": True # Show animation only on hover
                        }
                    },
                    "data": lines_data
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
        st_echarts(options=option, height="650px") # Increased height for more vertical space
        st.markdown('</div>', unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
        import traceback
        st.error(traceback.format_exc())
        st.error("Please ensure your observation sequence contains only integers separated by commas.")
