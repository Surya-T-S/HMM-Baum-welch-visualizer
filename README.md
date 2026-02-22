# Hidden Markov Model using Baum–Welch Algorithm

## Name:
Surya T S

## University Register Number:
TCR24CS069

## Description:
This project implements a Hidden Markov Model (HMM) trained using the Baum–Welch (Expectation-Maximization) algorithm using the hmmlearn library in Python.

The program:
- Takes a discrete observation sequence
- Trains an HMM
- Computes:
  - Initial distribution (pi)
  - Transition matrix (A)
  - Emission matrix (B)
  - Log likelihood P(O | lambda)
- Plots log likelihood vs iterations

## Installation:
pip install -r requirements.txt

## How to Run the Web App:
streamlit run app.py

## How to Run the CLI Script:
python train.py

## Example Output:
- Initial distribution
- Transition matrix
- Emission matrix
- Log likelihood
- Convergence graph
