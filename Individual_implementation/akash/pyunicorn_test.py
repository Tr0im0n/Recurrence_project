import numpy as np
from scipy.integrate import solve_ivp
from pyunicorn.timeseries.recurrence_plot import RecurrencePlot

# Define the Lorenz system of equations
def lorenz(t, state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    return [sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z]

# Time points for integration
t_span = (0, 100)
t_eval = np.linspace(*t_span, 10000)  # Evaluate the solution at these time points

# Initial conditions [x0, y0, z0]
initial_conditions = [1.0, 0.0, 0.0]

# Integrate the Lorenz system
sol = solve_ivp(lorenz, t_span, initial_conditions, t_eval=t_eval, method='RK45')

# Extract the time series from the solution
time_series = sol.y[0]  # Choose x component of the Lorenz attractor

# Calculate a suitable threshold for recurrence plot (e.g., using a fraction of the standard deviation)
threshold = 0.1 * np.std(time_series)

# Create an instance of RecurrencePlot with the Lorenz attractor time series and the threshold
rp = RecurrencePlot(time_series, threshold=threshold)

# Compute RQA measures
rr = rp.recurrence_rate()
det = rp.determinism()
l_max = rp.max_diaglength()
lam = rp.laminarity()

# Print results
print(f"Recurrence Rate (RR): {rr}")
print(f"Determinism (DET): {det}")
print(f"Max diagonal line length (L_max): {l_max}")
print(f"Laminarity (LAM): {lam}")
