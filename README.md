# Hidden Markov Model (HMM) - Baum-Welch Algorithm

A minimal, interactive implementation of the Baum-Welch (Expectation-Maximization) algorithm for training Hidden Markov Models. Built with Python, `hmmlearn`, and `Streamlit`.

---

## Author Details
- **Name:** Surya T S
- **University Register Number:** TCR24CS069

---

## Quick Start

### 1. Setup the Environment
It's recommended to use the included virtual environment to keep dependencies isolated.

```powershell
# Activate the virtual environment (Windows)
.\venv\Scripts\activate

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

### 2. Run the Interactive Web App (Recommended)
Experience the algorithm visually with dynamic inputs, matrix rendering, and highly accurate state transition diagrams.

```powershell
python -m streamlit run app.py
```

### 3. Run the CLI Version
For a quick terminal-based execution with matplotlib convergence plots:

```powershell
python train.py
```

---

## Gallery

*(Add your screenshots and videos here!)*

### Web App Interface
> **How to add an image:** 
> 1. Take a screenshot of your Streamlit app running.
> 2. Save it in this folder as `app_screenshot.png`.
> 3. Replace this text with: `![Web App Interface](./app_screenshot.png)`

### State Transition Diagram
> **How to add an image:** 
> 1. Save the generated Graphviz diagram as `diagram.png`.
> 2. Replace this text with: `![State Transition Diagram](./diagram.png)`

### Demo Video
> **How to add a video:**
> 1. Record a short screen capture of you using the app (e.g., using OBS or Windows Game Bar).
> 2. Save it as `demo.mp4` or upload it to YouTube/GitHub.
> 3. If uploaded to GitHub, you can link it like this: `[Watch the Demo Video](./demo.mp4)`
> 4. If on YouTube: `[![Demo Video](https://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)`

---

## What it Computes
- **Initial Distribution ($\pi$)**
- **Transition Matrix ($A$)**
- **Emission Matrix ($B$)**
- **Log Likelihood $P(O | \lambda)$**
- **Convergence Graph** (Log likelihood vs Iterations)
