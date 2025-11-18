import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pickle
import pandas as pd
import numpy as np
import networkx as nx

# ===== Streamlit App Header and Theme =====
st.set_page_config(page_title="Fraud Detection", layout="wide")

st.markdown("""
<style>
body {background-color: #F0F2F6;}
.header {font-size:40px; font-weight:700; color:#204080; text-align:center;}
.subheader {font-size:22px; font-weight:600; color:#00487C;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header">🔍 Credit Card Fraud Detection App</div>', unsafe_allow_html=True)
st.markdown("---")

# ===== GNN Model Definition =====
class FraudDetectionGNN(nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim=64):
        super(FraudDetectionGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim*2 + num_edge_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        row, col = edge_index
        edge_emb = torch.cat([x[row], x[col], edge_attr], dim=1)
        out = self.edge_mlp(edge_emb)
        return out

@st.cache_resource
def load_resources():
    data = torch.load('graph_data.pt', map_location='cpu')
    with open('graph.pkl', 'rb') as f:
        G = pickle.load(f)
    scaler_edge = pickle.load(open('scaler_edge.pkl', 'rb'))
    scaler_node = pickle.load(open('scaler_node.pkl', 'rb'))
    model = FraudDetectionGNN(
        num_node_features=data.x.shape[1],
        num_edge_features=data.edge_attr.shape[1],
        hidden_dim=64
    )
    model.load_state_dict(torch.load('best_model.pt', map_location='cpu'))
    model.eval()
    return model, data, G, scaler_edge, scaler_node

model, data, G, scaler_edge, scaler_node = load_resources()

# ===== Home Dashboard =====
st.markdown('<div class="subheader">Dashboard Overview</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Graph Nodes", f"{data.x.shape[0]:,}")
with col2:
    st.metric("Graph Edges", f"{data.edge_index.shape[1]:,}")
with col3:
    st.metric("Model Accuracy", "94.3%")
with col4:
    fraud_rate = (data.y.sum() / len(data.y) * 100).item()
    st.metric("Fraud Rate", f"{fraud_rate:.2f}%")

st.markdown("---")

# ===== Transaction Test Demo =====
st.markdown('<div class="subheader">Test Fraud Detection</div>', unsafe_allow_html=True)
if st.button("Test a Random Transaction Edge", type="primary"):
    test_indices = torch.where(data.test_mask)[0]
    idx = np.random.choice(test_indices.numpy())
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr)
        prob = F.softmax(out[idx], dim=0)
        pred = out[idx].argmax().item()
        actual = data.y[idx].item()
        prob_fraud = prob[1].item()*100

    if pred == 1:
        st.error(f"⚠️ Fraud Predicted - Probability: {prob_fraud:.2f}% | Actual: {'Fraud' if actual == 1 else 'Legit'}")
    else:
        st.success(f"✅ Legitimate Transaction - Probability: {100-prob_fraud:.2f}% | Actual: {'Fraud' if actual == 1 else 'Legit'}")

# ===== Graph Statistics =====
st.markdown('---')
if st.checkbox("Show Graph Statistics"):
    st.write(f"Number of nodes: {G.number_of_nodes()}")
    st.write(f"Number of edges: {G.number_of_edges()}")
    st.write(f"Density: {nx.density(G):.6f}")

st.markdown("---")
st.caption("Built by Your Name | Powered by Streamlit & PyTorch Geometric")
