import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def plot_predictions(y_true, y_pred, title="Comparing Input and ML Predicted Average Fidelity (with unitary gauge)"):
    """
    Plots predicted vs true values with scatter, 1:1 line, and linear fit.
    Legend is placed outside the plot on the right.
    """
    # Scatter plot
    plt.figure(figsize=(8,6))
    plt.scatter(y_true, y_pred, alpha=0.3, label='Predictions')

    # Ideal 1:1 line
    xmin, xmax = min(y_true), max(y_true)
    plt.plot([xmin, xmax], [xmin, xmax], linestyle='--', color='C1', label='Ideal 1:1 Line')

    # Linear fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)
    x_fit = np.linspace(xmin, xmax, 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, color='C3',label=f'Fit: y = {slope:.3f}x + {intercept:.3f}\nR² = {r_value**2:.4f}')

    # Labels and title
    plt.xlabel("Input Fidelity", fontsize=14)
    plt.ylabel("Predicted Fidelity", fontsize=14)
    plt.title(title, fontsize=16)

    # Grid
    plt.grid(True)

    # Legend below the plot
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
            ncol=3, fontsize=12)  # ncol=3 spreads legend items in 3 columns

    # Adjust layout so nothing is cut off
    plt.tight_layout()


    plt.show()
