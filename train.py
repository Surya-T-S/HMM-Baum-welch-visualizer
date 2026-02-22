# STEP 1: Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import MultinomialHMM
import warnings
import logging

# STEP 2: Define observation sequence
observations = [0, 1, 1, 0, 1, 0, 1]
# Convert it into numpy format required by hmmlearn
X = np.array(observations).reshape(-1, 1)

# STEP 3: Define number of hidden states
n_states = 2

# STEP 4: Create HMM model
# We set n_iter=1 because we will manually run multiple iterations
# to store log-likelihood values per iteration.
# Suppress hmmlearn warnings
warnings.filterwarnings("ignore")
logging.getLogger("hmmlearn").setLevel(logging.CRITICAL)

model = MultinomialHMM(
    n_components=n_states,
    n_iter=1,
    tol=1e-4,
    init_params="ste"
)

# STEP 5: Initialize log likelihood history list
log_likelihoods = []

# STEP 6: Manually run Baum–Welch training loop
max_iterations = 50

for i in range(max_iterations):
    # Call model.fit(X)
    model.fit(X)
    # Compute log likelihood using model.score(X)
    score = model.score(X)
    # Append score to log_likelihoods
    log_likelihoods.append(score)

# STEP 7: After training completes, print clearly
print("Final Initial Distribution (pi):")
print(model.startprob_)

print("\nFinal Transition Matrix (A):")
print(model.transmat_)

print("\nFinal Emission Matrix (B):")
print(model.emissionprob_)

print("\nFinal Log Likelihood P(O | lambda):")
print(model.score(X))

# STEP 8: Plot convergence graph
plt.figure(figsize=(12, 5))

# Plot 1: Log Likelihood vs Iterations
plt.subplot(1, 2, 1)
plt.plot(range(1, max_iterations + 1), log_likelihoods, marker='o', linestyle='-')
plt.title("Log Likelihood P(O | lambda) vs Iterations")
plt.xlabel("Iteration")
plt.ylabel("Log Likelihood")
plt.grid(True)

# Plot 2: 1 - P(O | lambda) vs Iterations (using exp to get probability from log likelihood)
# Note: Since log likelihoods are negative, exp(log_likelihood) gives the probability
probabilities = np.exp(log_likelihoods)
one_minus_probs = 1 - probabilities

plt.subplot(1, 2, 2)
plt.plot(range(1, max_iterations + 1), one_minus_probs, marker='x', linestyle='-', color='red')
plt.title("1 - P(O | lambda) vs Iterations")
plt.xlabel("Iteration")
plt.ylabel("1 - Probability")
plt.grid(True)

plt.tight_layout()
plt.show()

# Optional: State Transition Diagram using networkx (if installed)
try:
    import networkx as nx
    
    G = nx.DiGraph()
    
    # Add nodes
    for i in range(n_states):
        G.add_node(f"State {i}")
        
    # Add edges with transition probabilities
    for i in range(n_states):
        for j in range(n_states):
            prob = model.transmat_[i, j]
            if prob > 0:
                G.add_edge(f"State {i}", f"State {j}", weight=prob, label=f"{prob:.2f}")
                
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=2, arrowsize=20, connectionstyle="arc3,rad=0.1")
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")
    
    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    
    plt.title("State Transition Diagram")
    plt.axis('off')
    plt.show()
except ImportError:
    print("\nNote: Install 'networkx' to visualize the state transition diagram (pip install networkx).")
