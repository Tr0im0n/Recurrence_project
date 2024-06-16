from scipy.spatial.distance import pdist, squareform
from attractor_functions import *
from rqa_functions import *
import pandas as pd

class LivePlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Data and Recurrence Plot")

        # Configure tab system
        self.notebook = ttk.Notebook(root)
        self.home_tab = ttk.Frame(self.notebook)
        self.function_tab = ttk.Frame(self.notebook)
        self.data_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.home_tab, text='Main Menu')
        self.notebook.add(self.function_tab, text='Plotting Functions')
        self.notebook.add(self.data_tab, text='Plotting Data')
        self.notebook.pack(expand=1, fill='both')

        # Title label
        self.title_label = ttk.Label(self.home_tab, text="Welcome to Live Data and Recurrence Plot GUI",
                                     font=("Helvetica", 16))
        self.title_label.pack(pady=10)

        # Textbox explaining what the GUI does
        self.info_text = tk.Text(self.home_tab, wrap='word', height=10, width=50)
        self.info_text.insert(tk.END,
                              "This GUI allows you to visualize different chaotic systems and analyze their recurrence plots. "
                              "You can start/stop/reset the live plotting of selected functions and display histograms of recurrence quantification analysis (RQA) measures.")
        self.info_text.config(state=tk.DISABLED)  # Make the textbox read-only
        self.info_text.pack(pady=10)

        # Buttons to navigate to the other tabs
        self.btn_functions_tab = ttk.Button(self.home_tab, text="Go to Plotting Functions",
                                            command=lambda: self.notebook.select(self.function_tab))
        self.btn_functions_tab.pack(pady=5)

        self.btn_data_tab = ttk.Button(self.home_tab, text="Go to Plotting Data",
                                       command=lambda: self.notebook.select(self.data_tab))
        self.btn_data_tab.pack(pady=5)

        # Configure grid layout in the function_tab
        self.function_tab.columnconfigure(0, weight=1)
        self.function_tab.columnconfigure(1, weight=1)
        self.function_tab.rowconfigure(0, weight=1)
        self.function_tab.rowconfigure(1, weight=1)

        # Create a figure and three subplots
        self.fig_ps = plt.Figure()
        self.fig_comp = plt.Figure()
        self.fig_rp = plt.Figure()

        # Embedding the Matplotlib figures into Tkinter canvas
        self.canvas_ps = FigureCanvasTkAgg(self.fig_ps, master=self.function_tab)
        self.canvas_ps.get_tk_widget().grid(row=0, column=1, sticky='nsew')

        self.canvas_comp = FigureCanvasTkAgg(self.fig_comp, master=self.function_tab)
        self.canvas_comp.get_tk_widget().grid(row=1, column=0, sticky='nsew')

        self.canvas_rp = FigureCanvasTkAgg(self.fig_rp, master=self.function_tab)
        self.canvas_rp.get_tk_widget().grid(row=1, column=1, sticky='nsew')

        # Make frame for buttons
        self.command_window = ttk.Frame(self.function_tab)
        self.command_window.grid(row=0, column=0, sticky='nsew')

        self.command_window.columnconfigure(0, weight=1)
        self.command_window.columnconfigure(1, weight=1)
        self.command_window.columnconfigure(2, weight=1)
        self.command_window.rowconfigure(0, weight=1)
        self.command_window.rowconfigure(1, weight=1)
        self.command_window.rowconfigure(2, weight=1)
        self.command_window.rowconfigure(3, weight=1)

        # Add Start, Stop, and Reset buttons to control the live plotting
        self.btn_start = ttk.Button(self.command_window, text="Start", command=self.start)
        self.btn_start.grid(row=0, column=0)

        self.btn_stop = ttk.Button(self.command_window, text="Stop", command=self.stop)
        self.btn_stop.grid(row=1, column=0)

        self.btn_reset = ttk.Button(self.command_window, text="Reset", command=self.reset)
        self.btn_reset.grid(row=2, column=0)

        # Add button to open histogram window
        self.btn_histogram = ttk.Button(self.command_window, text="Show Histogram", command=lambda: show_histogram(self))
        self.btn_histogram.grid(row=3, column=0)

        # Add drop down menu to select function
        self.selected_option = tk.StringVar()
        functions = ["Lorenz", "Chua", "Rossler", "Chen"]
        self.selected_option.set(functions[0])
        self.dropdown = ttk.OptionMenu(self.command_window, self.selected_option, *functions)
        self.dropdown.grid(row=0, column=1)
        self.selected_option.trace("w", self.on_select)

        # Label to display RQA measures
        self.rqa_label = ttk.Label(self.command_window, text="RQA Measures will appear here")
        self.rqa_label.grid(row=0, column=2, rowspan=2)

        # Variables to control the live plotting
        self.is_running = False
        self.xyzs = np.array([[0., 1., 1.05]])  # Initialize xyzs with an initial value
        self.time_step = 0
        self.dt = 0.01


    def toggle_buttons(self, state):
            # Toggle all buttons except start and stop
            self.btn_reset.config(state=state)
            self.btn_histogram.config(state=state)
            self.dropdown.config(state=state)

    def plot_data(self):
        file_path = 'sample_data.csv'
        data_all = pd.read_csv(file_path, sep='\s+')
        data = data_all['SOI']
        print(data)

    def update_plot(self):
        if not self.is_running:
            return
        selected_function_name = self.selected_option.get().lower()
        if selected_function_name == 'lorenz':
            new_point = self.xyzs[-1] + lorenz(self.xyzs[-1]) * self.dt
        elif selected_function_name == 'chua':
            new_point = self.xyzs[-1] + chua(self.xyzs[-1]) * self.dt
        else:
            new_point = self.xyzs[-1]  # Default, no change

        self.xyzs = np.append(self.xyzs, [new_point], axis=0)
        self.fig_ps.clear()
        ax_ps = self.fig_ps.add_subplot(111, projection='3d')
        ax_ps.plot(*self.xyzs.T, lw=0.5)
        ax_ps.set_title(f"{selected_function_name.capitalize()} Attractor")
        ax_ps.set_xlabel("X Axis")
        ax_ps.set_ylabel("Y Axis")
        ax_ps.set_zlabel("Z Axis")

        self.x = [coord[0] for coord in self.xyzs]
        self.fig_comp.clear()
        ax_comp = self.fig_comp.add_subplot(111)
        ax_comp.plot(self.x, lw=0.5)
        ax_comp.set_title("X-coordinate")
        ax_comp.set_xlabel("Time")
        ax_comp.set_ylabel("X-coordinate")

        if len(self.xyzs) > 0:
            self.update_recurrence_plot()

        self.canvas_ps.draw()
        self.canvas_comp.draw()
        self.canvas_rp.draw()

        if self.is_running:
            self.root.after(10, self.update_plot)

    def update_recurrence_plot(self):
        self.fig_rp.clear()
        ax_rp = self.fig_rp.add_subplot(111)

        m = 10
        T = 3
        num_vectors = len(self.x) - (m - 1) * T
        vectors = np.array([self.x[t:t + m * T:T] for t in range(num_vectors)])
        if vectors.size > 0:
            D = squareform(pdist(vectors, metric='euclidean'))
            D_max = np.max(D)
            epsilon = 0.1 * D_max
            recurrence_matrix = D < epsilon
            x_rec, y_rec = np.argwhere(recurrence_matrix).T
            ax_rp.scatter(x_rec, y_rec, s=1)
            ax_rp.set_title("Recurrence Plot")
            ax_rp.set_xlabel("Vector Index")
            ax_rp.set_ylabel("Vector Index")
            rqa_measures = calculate_rqa_measures_pyrqa(self, vectors, epsilon)
            #det2, lam2 = self.calculate_manual_det_lam(recurrence_matrix)
            #rqa_measures["DET2"] = det2
            #rqa_measures["LAM2"] = lam2
            display_rqa_measures(self, rqa_measures)


    def start(self):
        if not self.is_running:
            self.is_running = True
            self.toggle_buttons("disabled")
            self.update_plot()

    def stop(self):
        if self.is_running:
            self.is_running = False
            self.toggle_buttons("normal")

    def reset(self):
        self.is_running = False
        self.xyzs = np.array([[0., 1., 1.05]])  # Reset the initial value
        self.fig_ps.clear()
        self.fig_comp.clear()
        self.fig_rp.clear()
        self.canvas_ps.draw()
        self.canvas_comp.draw()
        self.canvas_rp.draw()
        self.selected_option.set("Lorenz")

    def on_select(self, *args):
        self.reset()

if __name__ == "__main__":
    root = tk.Tk()
    app = LivePlotApp(root)
    root.mainloop()
