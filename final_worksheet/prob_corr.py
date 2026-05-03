import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate datasets
n = 100

# Strong positive linear correlation
x1 = np.linspace(0, 10, n)
y1 = 2 * x1 + np.random.normal(0, 2, n)

# Strong negative linear correlation
x2 = np.linspace(0, 10, n)
y2 = -2 * x2 + np.random.normal(0, 2, n)

# Weak/no correlation
x3 = np.linspace(0, 10, n)
y3 = np.random.normal(5, 5, n)

# Nonlinear correlation (parabolic)
#x4 = np.linspace(-3, 3, n)
#y4 = x4**2 + np.random.normal(0, 1, n)

# Nonlinear (sinusoidal relationship)
x5 = np.linspace(0, 4 * np.pi, n)
y5 = np.sin(x5) + np.random.normal(0, 0.2, n)

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(6, 6))
axs = axs.flatten()

# Plot each scatter plot
datasets = [(x1, y1, 'Fig. 1'), 
            (x2, y2, 'Fig. 2'),
            (x3, y3, 'Fig. 3'),
            #(x4, y4, 'Fig. 4'),
            (x5, y5, 'Fig. 4')]

for i, (x, y, title) in enumerate(datasets):
    axs[i].scatter(x, y, alpha=0.7)
    axs[i].set_title(title)
    axs[i].set_xlabel('X')
    axs[i].set_ylabel('Y')
    axs[i].grid(True)

# Hide the unused subplot
#axs[5].axis('off')

plt.savefig('prob_corr.png')

plt.tight_layout()
plt.show()
