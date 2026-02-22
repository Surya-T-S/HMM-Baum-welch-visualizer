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
        # INCREASED RADIUS significantly to spread states out much further
        radius = 180 
        center_x = 0
        center_y = 0
        
        # Distinct, vibrant color palette for each state to make transitions distinguishable
        state_colors = ['#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de', '#3ba272', '#fc8452', '#9a60b4', '#ea7ccc']
        
        for i in range(n_states):
            # Calculate angle for this node (evenly spaced around a circle)
            angle = (2 * math.pi * i) / n_states
            
            # Calculate x and y coordinates
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            
            color = state_colors[i % len(state_colors)]
            initial_prob = model.startprob_[i]
            
            nodes.append({
                "name": f"State {i}",
                "value": [x, y], # Coordinates for cartesian2d
                "tooltip": {
                    "formatter": f"<div style='text-align:center; padding:5px;'><b>State {i}</b><br/><hr style='margin:5px 0;border:none;border-top:1px solid #ccc;'/>Initial Probability: <br/><b style='font-size:16px; color:{color};'>{initial_prob:.2%}</b></div>"
                },
                "symbolSize": 100, # INCREASED NODE SIZE for better visibility
                "itemStyle": {
                    "color": "#ffffff", # Clean white background
                    "borderColor": color, # Unique color for each state
                    "borderWidth": 4, # Thicker border
                    "shadowBlur": 10, # Add drop shadow to nodes
                    "shadowColor": "rgba(0,0,0,0.2)"
                },
                "label": {
                    "show": True,
                    "fontSize": 18, # Larger font
                    "fontWeight": "bold",
                    "color": color, # Match border color
                    "position": "inside",
                    "formatter": "{b}" # Show name
                }
            })
            
        # Prepare edges and animation lines
        links = []
        lines_data = []
        
        for i in range(n_states):
            source_color = state_colors[i % len(state_colors)]
            for j in range(n_states):
                prob = model.transmat_[i, j]
                if prob > 0.01:
                    # Calculate curvature to prevent overlapping of mutual edges
                    if i == j:
                        curveness = 0.7 # Self loop (tighter curve to keep it compact)
                    else:
                        curveness = 0.25
                    
                    # 1. Static visible edge for the graph
                    links.append({
                        "source": f"State {i}",
                        "target": f"State {j}",
                        "value": float(prob),
                        "tooltip": {
                            "formatter": f"<div style='padding:5px;'><b>Transition</b><br/><hr style='margin:5px 0;border:none;border-top:1px solid #ccc;'/>From: <b>State {i}</b><br/>To: <b>State {j}</b><br/>Probability: <b style='font-size:14px; color:{source_color};'>{prob:.2%}</b> <span style='color:#888;'>({prob:.4f})</span></div>"
                        },
                        "label": {
                            "show": True,
                            "formatter": f"{prob:.2f}",
                            "fontSize": 14, 
                            "fontWeight": "bold",
                            "color": "#333", 
                            "backgroundColor": "#ffffff", 
                            "padding": [4, 6], 
                            "borderRadius": 4, 
                            "borderWidth": 1.5, 
                            "borderColor": source_color # Border matches the source state
                        },
                        "lineStyle": {
                            "width": max(2.5, prob * 10), # Thicker lines overall
                            "curveness": curveness,
                            "color": source_color, # Line color matches the source state
                            "opacity": 0.6 # Lower default opacity so overlapping lines don't create dark blobs
                        }
                    })
                    
                    # 2. Animated particle edge (only for transitions between different states)
                    if i != j:
                        # Calculate exact start and end points on the edge of the circle
                        dx = nodes[j]["value"][0] - nodes[i]["value"][0]
                        dy = nodes[j]["value"][1] - nodes[i]["value"][1]
                        dist = math.hypot(dx, dy)
                        
                        if dist > 0:
                            # Node radius is 50 (symbolSize is 100, so radius is 50)
                            # We add padding (55) so it stops right at the border
                            node_radius = 55
                            
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
                                    "color": source_color # Particle matches the source state color
                                }
                            })
                    
        # ECharts configuration
        option = {
            "title": {
                "text": "Interactive State Transitions (Drag background to pan, scroll to zoom)",
                "left": "center",
                "textStyle": {
                    "fontSize": 18,
                    "fontWeight": "bold",
                    "color": "#2C3E50"
                }
            },
            "tooltip": {
                "trigger": "item",
                "backgroundColor": "rgba(255, 255, 255, 0.95)",
                "borderColor": "#ccc",
                "borderWidth": 1,
                "textStyle": {
                    "color": "#333"
                }
            },
            "xAxis": {
                "show": False,
                "type": "value",
                "min": -radius * 1.8, # Increased bounds to prevent clipping
                "max": radius * 1.8
            },
            "yAxis": {
                "show": False,
                "type": "value",
                "min": -radius * 1.8,
                "max": radius * 1.8
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
                    "coordinateSystem": "cartesian2d", 
                    "symbolSize": 100, # Match new node size
                    "edgeSymbol": ["none", "arrow"],
                    "edgeSymbolSize": [0, 20], # Larger arrows
                    "data": nodes,
                    "links": links,
                    "emphasis": {
                        "focus": "adjacency",
                        "lineStyle": {
                            "width": 10, # Much thicker on hover
                            "opacity": 1, # Fully opaque on hover
                            "shadowBlur": 15,
                            "shadowColor": "rgba(0, 0, 0, 0.4)"
                        },
                        "label": {
                            "fontSize": 20,
                            "fontWeight": "bold",
                            "backgroundColor": "#ffffff",
                            "borderColor": "#1976D2", # Highlight border color
                            "borderWidth": 2
                        }
                    }
                },
                {
                    "type": "lines",
                    "coordinateSystem": "cartesian2d",
                    "zlevel": 2,
                    "effect": {
                        "show": False, 
                        "period": 3, # Slightly faster
                        "trailLength": 0.4, # Longer tail
                        "symbol": "circle", 
                        "symbolSize": 6 # Slightly larger particle
                    },
                    "lineStyle": {
                        "color": "transparent", 
                        "width": 0
                    },
                    "emphasis": {
                        "focus": "adjacency",
                        "effect": {
                            "show": True 
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
            border: 2px solid #E0E0E0;
            border-radius: 8px;
            padding: 15px;
            background-color: #FAFAFA;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="echarts-container">', unsafe_allow_html=True)
        st_echarts(options=option, height="700px") # Increased height for massive visibility
        st.markdown('</div>', unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
        import traceback
        st.error(traceback.format_exc())
        st.error("Please ensure your observation sequence contains only integers separated by commas.")
