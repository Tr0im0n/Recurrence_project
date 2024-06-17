from scipy.spatial.distance import pdist, squareform
from attractor_functions import *
from rqa_functions import *
import pandas as pd

class LivePlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Data and Recurrence Plot")

        self.create_tabs()
        self.create_home_tab()
        self.create_functions_tab()
        self.create_data_tab()


    def create_tabs(self):
        # Configure tab system
        self.notebook = ttk.Notebook(root)
        self.home_tab = ttk.Frame(self.notebook)
        self.function_tab = ttk.Frame(self.notebook)
        self.data_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.home_tab, text='Main Menu')
        self.notebook.add(self.function_tab, text='Plotting Functions')
        self.notebook.add(self.data_tab, text='Plotting Data')
        self.notebook.pack(expand=1, fill='both')

    def create_home_tab(self):
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

    def create_functions_tab(self):
        # Configure grid layout in the function_tab
        self.function_tab.columnconfigure(0, weight=1)
        self.function_tab.columnconfigure(1, weight=1)
        self.function_tab.rowconfigure(0, weight=1)
        self.function_tab.rowconfigure(1, weight=1)

        # Create a figure and three subplots
        self.fig_ps_func = plt.Figure()
        self.fig_comp_func = plt.Figure()
        self.fig_rp_func = plt.Figure()

        # Embedding the Matplotlib figures into Tkinter canvas
        self.canvas_ps_func = FigureCanvasTkAgg(self.fig_ps_func, master=self.function_tab)
        self.canvas_ps_func.get_tk_widget().grid(row=0, column=1, sticky='nsew')

        self.canvas_comp_func = FigureCanvasTkAgg(self.fig_comp_func, master=self.function_tab)
        self.canvas_comp_func.get_tk_widget().grid(row=1, column=0, sticky='nsew')

        self.canvas_rp_func = FigureCanvasTkAgg(self.fig_rp_func, master=self.function_tab)
        self.canvas_rp_func.get_tk_widget().grid(row=1, column=1, sticky='nsew')

        # Make frame for buttons
        self.command_window_func = ttk.Frame(self.function_tab)
        self.command_window_func.grid(row=0, column=0, sticky='nsew')

        self.command_window_func.columnconfigure(0, weight=1)
        self.command_window_func.columnconfigure(1, weight=1)
        self.command_window_func.columnconfigure(2, weight=1)
        self.command_window_func.rowconfigure(0, weight=1)
        self.command_window_func.rowconfigure(1, weight=1)
        self.command_window_func.rowconfigure(2, weight=1)
        self.command_window_func.rowconfigure(3, weight=1)

        # Add Start, Stop, and Reset buttons to control the live plotting
        self.btn_start_func = ttk.Button(self.command_window_func, text="Start", command=self.start_func)
        self.btn_start_func.grid(row=0, column=0)

        self.btn_stop_func = ttk.Button(self.command_window_func, text="Stop", command=self.stop_func)
        self.btn_stop_func.grid(row=1, column=0)

        self.btn_reset_func = ttk.Button(self.command_window_func, text="Reset", command=self.reset_func)
        self.btn_reset_func.grid(row=2, column=0)

        # Add button to open histogram window
        self.btn_histogram_func = ttk.Button(self.command_window_func, text="Show Histogram",
                                             command=lambda: show_histogram(self))
        self.btn_histogram_func.grid(row=3, column=0)

        # Add drop down menu to select function
        self.selected_option_func = tk.StringVar()
        functions = ["Lorenz", "Chua", "Rossler", "Chen"]
        self.selected_option_func.set(functions[0])
        self.dropdown_func = ttk.OptionMenu(self.command_window_func, self.selected_option_func, *functions)
        self.dropdown_func.grid(row=0, column=1)
        self.selected_option_func.trace("w", self.on_select)

        # Label to display RQA measures
        self.rqa_label_func = ttk.Label(self.command_window_func, text="RQA Measures will appear here")
        self.rqa_label_func.grid(row=0, column=2, rowspan=2)

        # Variables to control the live plotting
        self.is_running = False
        self.xyzs = np.array([[0., 1., 1.05]])  # Initialize xyzs with an initial value
        self.time_step = 0
        self.dt = 0.01

    def create_data_tab(self):
        # Configure grid layout in the function_tab
        self.data_tab.columnconfigure(0, weight=1)
        self.data_tab.columnconfigure(1, weight=1)
        # self.data_tab.columnconfigure(2, weight=1)
        self.data_tab.rowconfigure(0, weight=1)
        self.data_tab.rowconfigure(1, weight=1)

        # Create a figure and three subplots
        self.fig_data = plt.Figure()
        self.fig_rp_data = plt.Figure()

        # Embedding the Matplotlib figures into Tkinter canvas
        self.canvas_comp_data = FigureCanvasTkAgg(self.fig_data, master=self.data_tab)
        self.canvas_comp_data.get_tk_widget().grid(row=1, column=0, sticky='nsew')

        self.canvas_rp_data = FigureCanvasTkAgg(self.fig_rp_data, master=self.data_tab)
        self.canvas_rp_data.get_tk_widget().grid(row=1, column=1, sticky='nsew')

        # Make frame for buttons
        self.command_window_data = ttk.Frame(self.data_tab)
        self.command_window_data.grid(row=0, column=0, columnspan=2, sticky='nsew')

        self.command_window_data.columnconfigure(0, weight=1)
        self.command_window_data.columnconfigure(1, weight=1)
        self.command_window_data.columnconfigure(2, weight=1)
        self.command_window_data.columnconfigure(3, weight=1)

        self.command_window_data.rowconfigure(0, weight=1)
        self.command_window_data.rowconfigure(1, weight=1)
        self.command_window_data.rowconfigure(2, weight=1)
        self.command_window_data.rowconfigure(3, weight=1)

        # Add Start, Stop, and Reset buttons to control the live plotting
        self.btn_run_data = ttk.Button(self.command_window_data, text="Run", command=self.run_data)
        self.btn_run_data.grid(row=0, column=0)

        self.btn_reset_data = ttk.Button(self.command_window_data, text="Reset", command=self.reset_data)
        self.btn_reset_data.grid(row=1, column=0)

        # Add button to open histogram window
        self.btn_histogram_data = ttk.Button(self.command_window_data, text="Show Histogram",
                                          command=lambda: show_histogram(self))
        self.btn_histogram_data.grid(row=3, column=0)

        # Add text inputs for embedding dimension (m), time delay (T), and threshold (e)
        self.embedding_dim_var = tk.IntVar(value=1)  # Initial value for m
        self.time_delay_var = tk.IntVar(value=1)  # Initial value for T
        self.threshold_var = tk.DoubleVar(value=1)  # Initial value for e

        self.embedding_dim_label = ttk.Label(self.command_window_data, text="Embedding Dimension (m):")
        self.embedding_dim_label.grid(row=0, column=1, sticky='e')
        self.embedding_dim_input = ttk.Entry(self.command_window_data, textvariable=self.embedding_dim_var)
        self.embedding_dim_input.grid(row=0, column=2, sticky='w')

        self.time_delay_label = ttk.Label(self.command_window_data, text="Time Delay (T):")
        self.time_delay_label.grid(row=1, column=1, sticky='e')
        self.time_delay_input = ttk.Entry(self.command_window_data, textvariable=self.time_delay_var)
        self.time_delay_input.grid(row=1, column=2, sticky='w')

        self.threshold_label = ttk.Label(self.command_window_data, text="Threshold (e):")
        self.threshold_label.grid(row=2, column=1, sticky='e')
        self.threshold_input = ttk.Entry(self.command_window_data, textvariable=self.threshold_var)
        self.threshold_input.grid(row=2, column=2, sticky='w')

        # Label to display RQA measures
        self.rqa_label_data = ttk.Label(self.command_window_data, text="RQA Measures will appear here")
        self.rqa_label_data.grid(row=0, column=3, rowspan=2)


    def toggle_buttons(self, state):
            # Toggle all buttons except start and stop
            self.btn_reset_func.config(state=state)
            self.btn_histogram_func.config(state=state)
            self.dropdown_func.config(state=state)

    def plot_data(self):
        if not self.is_running:
            return

        file_path = 'vibration_data_synthetic.csv'
        self.data_np = pd.read_csv(file_path, sep='\s+')
        self.data_full = self.data_np.to_numpy().reshape(-1,1)
        self.data = self.data_full[:1000]

        m = int(self.embedding_dim_var.get())
        T = int(self.time_delay_var.get())
        epsilon = int(self.threshold_var.get())

        # Construct embedded vectors
        num_vectors = len(self.data) - (m - 1) * T
        vectors = np.array([self.data[t:t + m * T:T] for t in range(num_vectors)]).reshape(-1,1)

        # Find recurrence points using distance between embedded vectors
        self.D = squareform(pdist(vectors, metric='euclidean'))
        hit = np.argwhere(self.D < epsilon)

        # Extract x and y coordinates of points of recurrence
        x_rec, y_rec = hit[:, 0], hit[:, 1]

        # Plot data
        self.fig_data.clear()
        ax_data = self.fig_data.add_subplot(111)
        ax_data.plot(self.data, lw=0.5)
        ax_data.set_title("Visualization of Inputted Data")
        ax_data.set_xlabel("X Axis")
        ax_data.set_ylabel("Y Axis")
        self.canvas_comp_data.draw()

        # Plot recurrence plot
        self.fig_rp_data.clear()
        ax_rp_data = self.fig_rp_data.add_subplot(111)
        ax_rp_data.scatter(x_rec, y_rec, s=1)
        ax_rp_data.set_title("Recurrence Plot")
        ax_rp_data.set_xlabel("Vector Index")
        ax_rp_data.set_ylabel("Vector Index")
        self.canvas_rp_data.draw()

        # Calculate and display RQA measures
        rqa_measures_data = calculate_rqa_measures_pyrqa(self, vectors, epsilon)
        # det2, lam2 = self.calculate_manual_det_lam(recurrence_matrix)
        # rqa_measures["DET2"] = det2
        # rqa_measures["LAM2"] = lam2
        display_rqa_measures(self, self.rqa_label_data, rqa_measures_data)

    def update_plot(self):
        if not self.is_running:
            return
        selected_function_name = self.selected_option_func.get().lower()
        if selected_function_name == 'lorenz':
            new_point = self.xyzs[-1] + lorenz(self.xyzs[-1]) * self.dt
        elif selected_function_name == 'chua':
            new_point = self.xyzs[-1] + chua(self.xyzs[-1]) * self.dt
        else:
            new_point = self.xyzs[-1]  # Default, no change

        self.xyzs = np.append(self.xyzs, [new_point], axis=0)
        self.fig_ps_func.clear()
        ax_ps = self.fig_ps_func.add_subplot(111, projection='3d')
        ax_ps.plot(*self.xyzs.T, lw=0.5)
        ax_ps.set_title(f"{selected_function_name.capitalize()} Attractor")
        ax_ps.set_xlabel("X Axis")
        ax_ps.set_ylabel("Y Axis")
        ax_ps.set_zlabel("Z Axis")

        self.x = [coord[0] for coord in self.xyzs]
        self.fig_comp_func.clear()
        ax_comp = self.fig_comp_func.add_subplot(111)
        ax_comp.plot(self.x, lw=0.5)
        ax_comp.set_title("X-coordinate")
        ax_comp.set_xlabel("Time")
        ax_comp.set_ylabel("X-coordinate")

        if len(self.xyzs) > 0:
            self.update_recurrence_plot_figure()

        self.canvas_ps_func.draw()
        self.canvas_comp_func.draw()
        self.canvas_rp_func.draw()

        if self.is_running:
            self.root.after(1, self.update_plot)

    def update_recurrence_plot_figure(self):
        self.fig_rp_func.clear()
        ax_rp_func = self.fig_rp_func.add_subplot(111)

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
            ax_rp_func.scatter(x_rec, y_rec, s=1)
            ax_rp_func.set_title("Recurrence Plot")
            ax_rp_func.set_xlabel("Vector Index")
            ax_rp_func.set_ylabel("Vector Index")
            rqa_measures_func = calculate_rqa_measures_pyrqa(self, vectors, epsilon)
            #det2, lam2 = self.calculate_manual_det_lam(recurrence_matrix)
            #rqa_measures["DET2"] = det2
            #rqa_measures["LAM2"] = lam2
            display_rqa_measures(self, self.rqa_label_func, rqa_measures_func)


    def start_func(self):
        if not self.is_running:
            self.is_running = True
            self.toggle_buttons("disabled")
            self.update_plot()

    def stop_func(self):
        if self.is_running:
            self.is_running = False
            self.toggle_buttons("normal")

    def reset_func(self):
        self.is_running = False
        self.xyzs = np.array([[0., 1., 1.05]])  # Reset the initial value
        self.fig_ps_func.clear()
        self.fig_comp_func.clear()
        self.fig_rp_func.clear()
        self.canvas_ps_func.draw()
        self.canvas_comp_func.draw()
        self.canvas_rp_func.draw()
        self.selected_option_func.set("Lorenz")

    def run_data(self):
        if not self.is_running:
            self.is_running = True
            self.plot_data()

    def reset_data(self):
        self.is_running = False
        self.fig_data.clear()
        self.fig_rp_data.clear()
        self.canvas_comp_data.draw()
        self.canvas_rp_data.draw()

    def on_select(self, *args):
        self.reset_func()

if __name__ == "__main__":
    root = tk.Tk()
    app = LivePlotApp(root)
    root.mainloop()
