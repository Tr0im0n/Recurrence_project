import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform


def recurrence_plot(data, m, T, epsilon):

    num_vectors = len(data) - (m - 1) * T
    vectors = np.array([data[t:t + m * T:T] for t in range(num_vectors)])
    D = squareform(pdist(vectors, metric='euclidean'))
    hit = np.argwhere(D < epsilon)
    R = np.where(D < epsilon, 1, 0)
    return R


class RecurrencePlotApp(tk.Tk):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.title("Recurrence Plot Explorer")

        self.embedding_dimension = tk.IntVar(value=2)
        self.time_delay = tk.IntVar(value=1)
        self.epsilon = tk.IntVar(value=1)

        self.create_widgets()
        self.plot_recurrence()

    def create_widgets(self):
        frame = ttk.Frame(self)
        frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(frame, text="Embedding Dimension:").pack(side=tk.LEFT)
        embedding_dimension_scale = ttk.Scale(frame, from_=1, to=10, variable=self.embedding_dimension, command=self.update_plot)
        embedding_dimension_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(frame, text="Time Delay:").pack(side=tk.LEFT)
        time_delay_scale = ttk.Scale(frame, from_=1, to=10, variable=self.time_delay, command=self.update_plot)
        time_delay_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(frame, text="Epsilon:").pack(side=tk.LEFT)
        epsilon_scale = ttk.Scale(frame, from_=1, to=100, variable=self.epsilon, command=self.update_plot)
        epsilon_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.figure = plt.Figure()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def plot_recurrence(self):
        embedding_dimension = self.embedding_dimension.get()
        time_delay = self.time_delay.get()
        epsilon = self.epsilon.get()/10
        distance_matrix = recurrence_plot(self.data, embedding_dimension, time_delay, epsilon)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        cax = ax.imshow(distance_matrix, cmap='binary', origin='lower')
        ax.set_title(f'Recurrence Plot (Embedding Dimension: {embedding_dimension}, Time Delay: {time_delay}), Epsilon: {epsilon}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Time')
        self.figure.colorbar(cax)

        self.canvas.draw()

    def update_plot(self, _):
        self.plot_recurrence()

# Sample data
data = np.sin(np.linspace(0, 4 * np.pi, 100))

app = RecurrencePlotApp(data)
app.mainloop()

