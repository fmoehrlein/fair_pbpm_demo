# Bias Mitigation for Predictive Business Process Monitoring

This tool enables bias mitigation in machine learning models used for predictive business process monitoring, with a focus on next activity prediction. It addresses biases based on sensitive attributes such as gender by leveraging knowledge distillation. The approach captures the decision structure of a neural network in a decision tree, which can be interpreted and manually modified to reduce bias. The neural network is then fine-tuned to align with the modified, less biased decision structure.
This method is based on the ideas presented in the paper **A Human-In-The-Loop Approach for Improving Fairness in Predictive Business Process Monitoring** from the **BPM 25** conference.

A live demo can be accessed at https://apps.pm.iisys.de/fair_pbpm.

## Get Started

### Upload
Upload an event log in the `.xes` format (maximum file size: 100MB).  
Alternatively, use the included simulated event log from the paper **A Human-In-The-Loop Approach for Improving Fairness in Predictive Business Process Monitoring**, which models a hospital cancer screening process and includes the bias-inducing attribute "gender".

### Train
Train a neural network on the uploaded or simulated event log data.

**Data Parameters:**
- **Prefix Length:** Number of previous activities used for next activity prediction.
- **Categorical Attributes:** Categorical case attributes present in the event log.
- **Numerical Attributes:** Numerical case attributes present in the event log.
- **Test Ratio:** Proportion of data used for evaluation instead of training.

> *Note: Currently, only case attributes are fully supported and tested.*

**Model Parameters:**
- **Epochs:** Number of training epochs.
- **Learning Rate:** Learning rate of the neural network.
- **Hidden Layers:** List describing the architecture of the network (e.g., `[64, 32]` for two hidden layers with 64 and 32 neurons respectively).

### Distill
Apply knowledge distillation by training a decision tree on the predictions made by the neural network.

**Decision Tree Parameters:**
- **Min Samples Split:** Minimum number of samples required to split an internal node.
- **Max Depth:** Maximum depth of the tree.
- **CCP Alpha:** Complexity parameter used for Minimal Cost-Complexity Pruning.

### Alterations
View and interactively modify the distilled decision tree.

**Modification Options:**
- **Retrain:** Retrain the subtree rooted at the selected node, disallowing splits on the node’s feature.
- **Remove:** Remove the selected node and replace it with one of its children. Choose manually (left/right) or automatically keep the subtree with more training samples.

### Fine Tuning
Fine tune the neural network using the modified decision tree.

**Finetuning Modes:**
- **All Samples:** Mimic decision tree predictions on all training samples.
- **Modified Samples:** Mimic tree predictions only on samples impacted by tree modifications, preserving original predictions elsewhere.

After fine tuning, the updated model can be distilled again using the same parameters as in the Distill step, allowing iterative refinement and inspection.
