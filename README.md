# Hidden Markov Model (HMM) - Baum-Welch Algorithm

A minimal, interactive implementation of the Baum-Welch (Expectation-Maximization) algorithm for training Hidden Markov Models. Built with Python, `hmmlearn`, `Plotly`, and `Apache ECharts` via `Streamlit`.

---

<div align="center">
  <h3><b>Author Details</b></h3>
  <p><b>Name:</b> Surya T S</p>
  <p><b>University Register Number:</b> TCR24CS069</p>
</div>

---

## Gallery

### Web App Interface
![Web App Interface](./assets/app_screenshot.png)

### Convergence Plots
![Convergence Plots](./assets/convergence_plots.png)

### State Transition Diagram
![State Transition Diagram](./assets/state_diagram.png)

---

## User Manual & Quick Start

### 1. Run the Interactive Web App (Recommended)
Experience the algorithm visually with dynamic inputs, matrix rendering, and highly accurate, animated state transition diagrams.

```powershell
python -m streamlit run app.py
```

#### How to use the Web App:
1. **Input Parameters**: On the left sidebar, enter the number of hidden states, the number of unique observations, and the maximum number of iterations for the Baum-Welch algorithm.
2. **Observation Sequence**: Enter a comma-separated list of integers representing your observation sequence (e.g., `0, 1, 0, 2, 1`). Ensure the numbers are between `0` and `Number of Observations - 1`.
3. **Train Model**: Click the "Train HMM Model" button.
4. **View Results**: 
   - The app will display the learned Initial Distribution ($\pi$), Transition Matrix ($A$), and Emission Matrix ($B$).
   - It will also show the Final Log Likelihood and the actual Probability $P(O | \lambda)$.
5. **Analyze Convergence**: Two interactive Plotly graphs will appear showing the Log Likelihood and $1 - P(O | \lambda)$ over the training iterations. You can hover, zoom, and pan these graphs.
6. **Explore State Transitions**: Scroll down to the interactive ECharts diagram. 
   - **Hover** over states to see their initial probabilities.
   - **Hover** over the dashed transition lines to highlight specific paths and see the exact transition probabilities.
   - **Drag** the background to pan around the diagram, and **scroll** your mouse wheel to zoom in and out.

### 2. Run the CLI Version
For a quick terminal-based execution that prints the matrices and final probabilities directly to the console:

```powershell
python train.py
```

---

## What it Computes
- **Initial Distribution ($\pi$)**
- **Transition Matrix ($A$)**
- **Emission Matrix ($B$)**
- **Log Likelihood $P(O | \lambda)$**
- **Convergence Graph** (Log likelihood vs Iterations)

---

## Installation & Setup

Follow these steps to download and run the project on your local machine.

### 1. Clone the Repository
First, download the project files to your computer using Git:
```powershell
git clone https://github.com/Surya-T-S/HMM-Baum-welch-visualizer.git
cd HMM-Baum-welch-visualizer
```
*(If you downloaded the project as a ZIP file, simply extract it and open the folder in your terminal).

### 2. Create a Virtual Environment (Recommended)
It is highly recommended to create a Python virtual environment to keep the project dependencies isolated from your system Python.

```powershell
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment (Windows)
.\venv\Scripts\activate

# Activate the virtual environment (Mac/Linux)
# source venv/bin/activate
```

### 3. Install Dependencies
With the virtual environment activated, install all required Python packages:
```powershell
pip install -r requirements.txt
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
