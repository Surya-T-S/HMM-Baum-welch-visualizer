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
        
        # Eye-pleasing, highly visible color palette (deep jewel tones and rich colors)
        state_colors = ["#2E86C1", "#28B463", "#D68910", "#CB4335", "#8E44AD", "#17A589", "#D4AC0D", "#BA4A00", "#2C3E50"]
        
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
                    "color": "#ffffff", # Pure white states
                    "borderColor": color, # Border matches the state color for better definition
                    "borderWidth": 4, # Thicker border for clarity
                    "shadowBlur": 15, 
                    "shadowColor": "rgba(0,0,0,0.2)", # Slightly stronger shadow
                    "shadowOffsetX": 0,
                    "shadowOffsetY": 4
                },
                "label": {
                    "show": True,
                    "fontSize": 20, # Larger font
                    "fontWeight": "bold",
                    "color": "#000000", # Pure black text for maximum readability
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
                            "formatter": f"<div style='padding:5px; text-align:center;'><b>Transition</b><br/><hr style='margin:5px 0;border:none;border-top:1px solid #ccc;'/><b>State {i}</b> &rarr; <b>State {j}</b><br/>Probability: <b style='font-size:14px; color:{source_color};'>{prob:.2%}</b> <span style='color:#888;'>({prob:.4f})</span></div>"
                        },
                        "label": {
                            "show": True,
                            "formatter": f"{prob:.2f}",
                            "fontSize": 16, # Larger font for clarity
                            "fontWeight": "bold", # Bold font for clarity
                            "color": "#000000", # Pure black text for maximum contrast
                            "backgroundColor": "#ffffff", 
                            "padding": [4, 6], # More padding
                            "borderRadius": 4, 
                            "borderWidth": 2, # Thicker border
                            "borderColor": source_color # Border matches the source state color
                        },
                        "lineStyle": {
                            "width": max(3.0, prob * 10), # Thicker lines for clarity
                            "curveness": curveness,
                            "color": source_color, # Line color matches the source state
                            "opacity": 0.95, # Very high opacity so lines are very clear and solid
                        }
                    })
                    
        # ECharts configuration
        option = {
            "title": {
                "text": "Interactive State Transitions (Drag background to pan, scroll to zoom)",
                "left": "center",
                "textStyle": {
                    "fontSize": 16,
                    "fontWeight": "bold", # Bolder title
                    "color": "#444444"
                }
            },
            "tooltip": {
                "trigger": "item",
                "backgroundColor": "rgba(255, 255, 255, 0.95)",
                "borderColor": "#cccccc", 
                "borderWidth": 1,
                "textStyle": {
                    "color": "#333333"
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
                    "edgeSymbolSize": [0, 15], # Smaller arrows as requested
                    "data": nodes,
                    "links": links,
                    "blur": {
                        "itemStyle": {
                            "opacity": 0.2 # Make non-adjacent nodes faint
                        },
                        "lineStyle": {
                            "opacity": 0.05 # Make non-adjacent lines almost invisible
                        }
                    },
                    "emphasis": {
                        "focus": "adjacency",
                        "lineStyle": {
                            "width": 5, # Explicitly set a width instead of inherit
                            "opacity": 1, # Fully opaque on hover
                            "shadowBlur": 0 # Remove shadow
                        },
                        "label": {
                            "fontSize": 16, # Keep original font size
                            "fontWeight": "bold",
                            "backgroundColor": "#ffffff",
                            "borderColor": "inherit", # Keep original border color
                            "borderWidth": 2 # Keep original border width
                        }
                    }
                }
            ]
        }
        
        # Add a nice border around the interactive diagram
        st.markdown("""
        <style>
        .echarts-container {
            border: 1px solid #f0f0f0;
            border-radius: 12px;
            padding: 15px;
            background-color: #ffffff;
            box-shadow: 0 4px 20px rgba(0,0,0,0.03);
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
