CNN_TN = [
    96.40,
    95.94,
    96.69,
    96.34,
    96.34,
    96.46,
    96.23,
    96.74,
    96.97,
    95.83,
    96.34,
    97.26,
    96.86,
    96.46,
    96.46,
    96.69,
]

CNN_TP = [
    98.40,
    98.51,
    98.86,
    99.14,
    98.91,
    98.80,
    99.54,
    99.09,
    98.91,
    99.14,
    98.74,
    98.97,
    98.74,
    98.74,
    98.40,
    98.63,
]

ViT_TN = [
    87.77,
    87.54,
    87.83,
    88.11,
    86.11,
    88.11,
    85.83,
    85.89,
    87.37,
    86.91,
    84.74,
    87.49,
    86.69,
    83.31,
    86.11,
    86.00,
]

ViT_TP = [
    91.54,
    92.69,
    90.46,
    94.06,
    92.23,
    92.29,
    92.69,
    92.46,
    91.37,
    91.43,
    90.34,
    90.63,
    91.31,
    91.43,
    88.91,
    90.74,
]

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

# --- LaTeX rendering and fonts ---
rc("text", usetex=True)
rc("font", family="serif", serif=["Roman"])
rc("axes", unicode_minus=False)

# # --- Data ---
# CNN_TN = [96.40, 95.94, 96.69, 96.34, 96.34, 96.46, 96.23, 96.74, 96.97, 95.83, 96.34, 97.26, 96.86, 96.46, 96.46, 96.69]
# CNN_TP = [98.40, 98.51, 98.86, 99.14, 98.91, 98.80, 99.54, 99.09, 98.91, 99.14, 98.74, 98.97, 98.74, 98.74, 98.40, 98.63]
# ViT_TN = [87.77, 87.54, 87.83, 88.11, 86.11, 88.11, 85.83, 85.89, 87.37, 86.91, 84.74, 87.49, 86.69, 83.31, 86.11, 86.00]
# ViT_TP = [91.54, 92.69, 90.46, 94.06, 92.23, 92.29, 92.69, 92.46, 91.37, 91.43, 90.34, 90.63, 91.31, 91.43, 88.91, 90.74]

# --- X-axis values ---
epochs = np.arange(1, len(CNN_TN) + 1)

# --- Font size controls ---
label_fontsize = 14
title_fontsize = 16
tick_fontsize = 12
legend_fontsize = 12

# --- Plot ---
plt.figure(figsize=(8, 5))
plt.plot(epochs, CNN_TP, "-", color="C0", label=r"\textbf{CNN* Unknots}", linewidth=1.8)
plt.plot(epochs, CNN_TN, "--", color="C0", label=r"\textbf{CNN* Knots}", linewidth=1.8)
plt.plot(epochs, ViT_TP, "-", color="C1", label=r"\textbf{ViT* Unknots}", linewidth=1.8)
plt.plot(epochs, ViT_TN, "--", color="C1", label=r"\textbf{ViT* Knots}", linewidth=1.8)


# --- Labels & Title ---
plt.xlabel(r"\textbf{Crossings}", fontsize=label_fontsize)
plt.ylabel(r"\textbf{Percent Correct}", fontsize=label_fontsize)
plt.title(r"\textbf{CNN* and ViT* Test Accuracy by Crossing Count}", fontsize=title_fontsize)


plt.yticks(np.arange(55, 101, 5))
plt.xticks(np.arange(1, 17), [str(i) for i in range(20, 36)], fontsize=tick_fontsize)

plt.grid(alpha=0.3, linestyle="--")
plt.legend(fontsize=legend_fontsize, loc="lower right")
plt.tight_layout()

# --- Save and show ---
plt.savefig("acc-by-nc.png", dpi=300, bbox_inches="tight")