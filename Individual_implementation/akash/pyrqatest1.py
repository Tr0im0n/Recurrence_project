import numpy as np
from scipy.integrate import solve_ivp
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation

# Define the Lorenz system of differential equations
def lorenz(t, xyz, sigma=10, rho=28, beta=8/3):
    x, y, z = xyz
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Generate Lorenz attractor data points
t_span = (0, 100)  # Time span for integration
t_eval = np.linspace(0, 100, 10000)  # Time points to evaluate solution
initial_conditions = [1.0, 1.0, 1.0]  # Initial conditions [x0, y0, z0]
sol = solve_ivp(lorenz, t_span, initial_conditions, t_eval=t_eval)

# Extract the x-component (or any component) as the time series data
lorenz_data_points = sol.y[0]  # Using x-component for simplicity

# Perform Recurrence Quantification Analysis (RQA)

# Create TimeSeries object
time_series = TimeSeries(lorenz_data_points,
                         embedding_dimension=3,  # Adjust embedding dimension as needed
                         time_delay=1)           # Adjust time delay as needed

# Create Settings object
settings = Settings(time_series,
                    analysis_type=Classic,
                    neighbourhood=FixedRadius(0.1),  # Adjust radius as needed
                    similarity_measure=EuclideanMetric,
                    theiler_corrector=1)

# Create RQAComputation object
computation = RQAComputation.create(settings,
                                    verbose=True)

# Run computation
result = computation.run()

# Set RQA parameters
result.min_diagonal_line_length = 2
result.min_vertical_line_length = 2
result.min_white_vertical_line_length = 2

# Print the RQA result
print(result)
