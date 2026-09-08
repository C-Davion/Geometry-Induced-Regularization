'''
Reads a .pt file containing the model weights of the class OneHiddenLayerReLUNEt, requires knowledge of which rho was used
Such file are generated with the training file train.py
After reading the .pt file, instantied the network and plots the output for a given x in range in R.
Piece with different activation pattern are colored differently.

'''

import torch
import matplotlib.pyplot as plt
from relu_network import OneHiddenLayerReLUNet, QuadraticFeatureMap
# ------------------------------------------------------------
# Import your network class
# ------------------------------------------------------------
# Change this import to wherever OneHiddenLayerReLUNet is defined.



# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
WEIGHTS_FILE = r"reg_experiments\run_11.36_h30_0903_3\model_weights.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

# Input range to visualize
X_MIN = -3.0
X_MAX = 3.0
N_POINTS = 300
input_dim=1
showbreakline=False #vertical lines between  activation pattern

# ------------------------------------------------------------
# Load trained weights
# ------------------------------------------------------------
checkpoint = torch.load(
    WEIGHTS_FILE,
    map_location=DEVICE,
    weights_only=False,
)

print("Checkpoint type:", type(checkpoint))

# ------------------------------------------------------------
# Reconstruct the network
# ------------------------------------------------------------
#
# Case 1:
#   model_weight.pt contains a state_dict:
#
#       torch.save(model.state_dict(), "model_weight.pt")
#
# In this case, you need to know the architecture parameters
# (W1/b1/W2/b2) to instantiate the model.
#
# Case 2:
#   model_weight.pt contains W1, b1, W2, b2 directly.
#
# Adjust the code below according to how you saved the model.
# ------------------------------------------------------------

if isinstance(checkpoint, dict):

    # If the checkpoint contains the individual weights
    if all(k in checkpoint for k in ["W1", "b1", "W2", "b2"]):

        W1 = checkpoint["W1"].to(device=DEVICE, dtype=DTYPE)
        b1 = checkpoint["b1"].to(device=DEVICE, dtype=DTYPE)
        W2 = checkpoint["W2"].to(device=DEVICE, dtype=DTYPE)
        b2 = checkpoint["b2"].to(device=DEVICE, dtype=DTYPE)

        # Change rho if your trained model used a different value.
        rho = QuadraticFeatureMap(input_dim).to(DEVICE,dtype=DTYPE)

        model = OneHiddenLayerReLUNet(
            W1, b1, W2, b2, rho=rho
        ).to(DEVICE, dtype=DTYPE)

    else:
        # If it is a normal state_dict, instantiate the model
        # using your known architecture here.
        #
        # Example:
        #
        # model = OneHiddenLayerReLUNet(
        #     W1, b1, W2, b2, rho=rho
        # ).to(DEVICE, dtype=DTYPE)
        #
        # model.load_state_dict(checkpoint)

        raise RuntimeError(
            "The file contains a state_dict, but the script does not "
            "know how to reconstruct W1, b1, W2 and b2.\n\n"
            "Modify the state_dict section of this script with your "
            "network architecture."
        )

else:
    raise RuntimeError(
        "Unknown model file format. Expected a dictionary/checkpoint."
    )


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------
model.eval()
print(model)
get_every_nth=N_POINTS//30
x = torch.linspace(
    X_MIN,
    X_MAX,
    N_POINTS,
    device=DEVICE,
    dtype=DTYPE,
)

# Shape: (N, 1)
x_input = x.reshape(-1, 1)

with torch.no_grad():
    y = model(x_input)
    patterns=model.activation_pattern(x_input)

# Remove unnecessary dimensions
y = y.squeeze()

# Move results back to CPU for matplotlib
x_np = x.detach().cpu().numpy()
y_np = y.detach().cpu().numpy()
patterns_np = patterns.detach().cpu().numpy()

pattern_tuples = [tuple(row.astype(int)) for row in patterns_np]

breaks = [0]
for i in range(1, len(pattern_tuples)):
    if pattern_tuples[i] != pattern_tuples[i - 1]:
        breaks.append(i)
breaks.append(len(x_np))

plt.figure(figsize=(10, 6))
cmap = plt.cm.get_cmap("tab20")

plt.scatter(x_np[::5],y_np[::5],marker='x')

for region in range(len(breaks) - 1):
    start = breaks[region]
    end = breaks[region + 1]

    start_plot = max(0, start - 1)

    plt.plot(
        x_np[start_plot:end],
        y_np[start_plot:end],
        color=cmap(region % 20),
        linewidth=2,
    )

    if start > 0 and showbreakline:
        plt.axvline(
            x_np[start],
            linestyle="--",
            linewidth=0.7,
            alpha=0.4,
        )

plt.xlabel("x")
plt.ylabel("f(x)")

plt.text(
    0.98, 0.02,
    f"Total number of patterns: {len(breaks)+1}",
    color="black",
    ha="right",
    va="bottom",
    transform=plt.gca().transAxes,
    bbox=dict(facecolor="none", edgecolor="black"),
)

plt.title("Network output colored by activation pattern")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()