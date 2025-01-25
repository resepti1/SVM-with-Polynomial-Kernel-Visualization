import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import tkinter as tk
import numpy as np
from scipy.interpolate import interp1d

from SVM_active_polynomial import SVM_polynomial

# grid for the decision boundary
x_min, x_max = 0, 10
y_min, y_max = 0, 10

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 50),
    np.linspace(y_min, y_max, 50)
)

grid_points = np.c_[xx.ravel(), yy.ravel()]

# Initialize the model and some variables
model = SVM_polynomial(1)

col = 'r'  # Color of the first point
X = []     # List for input points
y = [-1]   # List for labels (-1 or 1)
curves = []     # List for decision curves

def interpolate(t, y1, y2):
    """
    Linearly interpolate between two points or curves.
    
    Args:
        t (float): Interpolation factor (0 <= t <= 1).
        y1 (ndarray): Starting curve or point.
        y2 (ndarray): Ending curve or point.
    
    Returns:
        ndarray: Interpolated curve or point.
    """
    return (1 - t) * y1 + t * y2

def resample_curve(curve, num_points):
    """
    Resample a curve to ensure it has same number of points as the other one.
    
    Args:
        curve (ndarray): The input curve as an array of points.
        num_points (int): Desired number of points in the resampled curve.
    
    Returns:
        ndarray: Resampled curve.
    """
    # Compute cumulative distances along the curve
    distances = np.sqrt(np.sum(np.diff(curve, axis=0) ** 2, axis=1))
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

    # Create uniform distances for resampling
    uniform_distances = np.linspace(0, cumulative_distances[-1], num_points)

    # Interpolate x and y coordinates along the curve
    x_interp = interp1d(cumulative_distances,
                        curve[:, 0], kind='linear')(uniform_distances)
    y_interp = interp1d(cumulative_distances,
                        curve[:, 1], kind='linear')(uniform_distances)

    return np.vstack([x_interp, y_interp]).T

def on_click(event):
    """
    Handle mouse click events to add new points, update the decision boundary
    
    Args:
        event (MouseEvent): The mouse click event.
    """
    global col
    if event.inaxes is not None:
        # Add a new point
        X1, X2 = event.xdata, event.ydata
        X.append([X1, X2])

        # Update the decision boundary
        Z = model.get_graph(np.array(X), np.array(y), grid_points)
        Z = Z.reshape(xx.shape)

        if len(X) >= 2:
            # Extract contour curves for decision boundary
            current_curve = np.vstack([
                path.vertices for collection in plt.contour(
                    xx, yy, Z, levels=[0], colors="k", linewidths=2
                ).collections for path in collection.get_paths()
            ])
            curves.append(current_curve)

        # Clear previous plots
        for collection in ax.collections:
            collection.remove()

        # Plot updated decision boundary
        ax.contourf(xx, yy, Z, levels=[-1e6, 0, 1e6],
                    colors=['lightcoral', 'lightblue'], alpha=0.5)

        # Plot points
        for x, label in zip(X, y):
            color = 'r' if label == -1 else 'b'
            ax.scatter(x[0], x[1], c=color)

        if len(curves) >= 2:
            # Animate transition between the last two curves
            num_points = 100  # Number of points for interpolation
            curve1 = resample_curve(curves[-2], num_points)
            curve2 = resample_curve(curves[-1], num_points)

            def update(frame):
                """
                Update function for the animation.
                
                Args:
                    frame (int): Current frame index.
                
                Returns:
                    tuple: Updated line for the animation.
                """
                t = frame / 100  # Normalize frame index to [0, 1]
                x_interp = interpolate(t, curve1[:, 0], curve2[:, 0])
                y_interp = interpolate(t, curve1[:, 1], curve2[:, 1])

                line.set_data(x_interp, y_interp)
                return line,

            for line in ax.lines:
                line.remove()
            line, = ax.plot([], [], 'k-', lw=2)
            ani = FuncAnimation(fig, update, frames=101,
                                interval=1, blit=True, repeat=False)
        else:
            # Plot the current curve if less than two curves exist
            if len(X) >= 2:
                line, = ax.plot([], [], 'k-', lw=2)
                line.set_data(current_curve[:, 0], current_curve[:, 1])
        canvas.draw()

        # Toggle the label for the next point
        if y[-1] == -1:
            y.append(1)
            col = 'b'
        else:
            y.append(-1)
            col = 'r'

# Setup matplotlib figure
fig, ax = plt.subplots()
ax.set_title("Click to add new points")
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Setup Tkinter window
root = tk.Tk()
root.title("SVM Visualization")
root.resizable(False, False)

canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(fill=tk.BOTH, expand=1)

# Connect the click event to the on_click function
fig.canvas.mpl_connect('button_press_event', on_click)

# Start the Tkinter event loop
tk.mainloop()
