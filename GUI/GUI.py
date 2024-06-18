import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.spatial.distance import pdist, squareform
from attractor_functions import *
from rqa_functions import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter.filedialog as fd
from home_tab import homeTab


class LivePlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Data and Recurrence Plot")
        self.create_tabs()
        self.create_home_tab()
        # self.create_home_tab = homeTab(self.notebook)
        self.create_functions_tab()
        self.create_data_tab()

    def create_tabs(self):
        self.notebook = ttk.Notebook(root)
        self.home_tab = ttk.Frame(self.notebook)
        self.function_tab = ttk.Frame(self.notebook)
        self.data_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.home_tab, text='Main Menu')
        self.notebook.add(self.function_tab, text='Plotting Functions')
        self.notebook.add(self.data_tab, text='Plotting Data')
        self.notebook.pack(expand=1, fill='both')



    def create_functions_tab(self):
        self.function_tab.columnconfigure(0, weight=1)
        self.function_tab.columnconfigure(1, weight=1)
        self.function_tab.rowconfigure(0, weight=1)
        self.function_tab.rowconfigure(1, weight=1)

        self.fig_ps_func = plt.Figure()
        self.fig_comp_func = plt.Figure()
        self.fig_rp_func = plt.Figure()

        self.canvas_ps_func = FigureCanvasTkAgg(self.fig_ps_func, master=self.function_tab)
        self.canvas_ps_func.get_tk_widget().grid(row=0, column=1, sticky='nsew')

        self.canvas_comp_func = FigureCanvasTkAgg(self.fig_comp_func, master=self.function_tab)
        self.canvas_comp_func.get_tk_widget().grid(row=1, column=0, sticky='nsew')

        self.canvas_rp_func = FigureCanvasTkAgg(self.fig_rp_func, master=self.function_tab)
        self.canvas_rp_func.get_tk_widget().grid(row=1, column=1, sticky='nsew')

        self.command_window_func = ttk.Frame(self.function_tab)
        self.command_window_func.grid(row=0, column=0, sticky='nsew')

        self.command_window_func.columnconfigure(0, weight=1)
        self.command_window_func.columnconfigure(1, weight=2)
        self.command_window_func.columnconfigure(2, weight=1)
        self.command_window_func.rowconfigure(0, weight=1)
        self.command_window_func.rowconfigure(1, weight=1)
        self.command_window_func.rowconfigure(2, weight=1)
        self.command_window_func.rowconfigure(3, weight=1)

        self.btn_start_func = ttk.Button(self.command_window_func, text="Start", command=self.start_func)
        self.btn_start_func.grid(row=0, column=0)

        self.btn_stop_func = ttk.Button(self.command_window_func, text="Stop", command=self.stop_func)
        self.btn_stop_func.grid(row=1, column=0)

        self.btn_reset_func = ttk.Button(self.command_window_func, text="Reset", command=self.reset_func)
        self.btn_reset_func.grid(row=2, column=0)

        self.selected_option_func = tk.StringVar()
        functions = ["Lorenz", "Chua", "Rossler", "Chen"]
        self.selected_option_func.set(functions[0])
        self.dropdown_func = ttk.OptionMenu(self.command_window_func, self.selected_option_func, *functions)
        self.dropdown_func.grid(row=0, column=1)
        self.selected_option_func.trace("w", self.on_select)

        # Animation speed Control
        self.slider_window = ttk.Frame(self.command_window_func)
        self.slider_window.grid(row=1, column=1, sticky='nsew')

        self.time_step_var = tk.IntVar(value=1)
        self.slider_label = ttk.Label(self.slider_window, text="Animation Speed:")
        self.slider_label.pack()
        self.time_step_slider = ttk.Scale(self.slider_window, from_=1, to=10, variable=self.time_step_var,
                                          orient=tk.HORIZONTAL)
        self.time_step_slider.pack()

        # Set up for accepting user inputs
        self.init_cond_frame = tk.Frame(self.command_window_func)
        self.init_cond_frame.grid(row=2, column=1, sticky='nsew')

        self.command_window_func.columnconfigure(0, weight=1)
        self.command_window_func.columnconfigure(1, weight=1)
        self.command_window_func.columnconfigure(2, weight=1)
        self.command_window_func.columnconfigure(3, weight=1)
        self.command_window_func.rowconfigure(0, weight=1)
        self.command_window_func.rowconfigure(1, weight=1)
        self.command_window_func.rowconfigure(2, weight=1)
        self.command_window_func.rowconfigure(3, weight=1)

        # Initial Conditions
        self.init_cond_label = tk.Label(self.init_cond_frame, text='Initial Conditions:')
        self.init_cond_label.grid(row=0,column=0, columnspan=2)
        self.init_cond_x_label = tk.Label(self.init_cond_frame, text='X:')
        self.init_cond_x_label.grid(row=1, column=0)
        self.init_cond_y_label = tk.Label(self.init_cond_frame, text='Y:')
        self.init_cond_y_label.grid(row=2, column=0)
        self.init_cond_z_label = tk.Label(self.init_cond_frame, text='Z:')
        self.init_cond_z_label.grid(row=3, column=0)

        self.init_cond_x_var = tk.IntVar(value=1)
        self.init_cond_y_var = tk.IntVar(value=1)
        self.init_cond_z_var = tk.IntVar(value=1)

        self.init_cond_x_input = tk.Entry(self.init_cond_frame, textvariable=self.init_cond_x_var)
        self.init_cond_x_input.grid(row=1, column=1)
        self.init_cond_y_input = tk.Entry(self.init_cond_frame, textvariable=self.init_cond_y_var)
        self.init_cond_y_input.grid(row=2, column=1)
        self.init_cond_z_input = tk.Entry(self.init_cond_frame, textvariable=self.init_cond_z_var)
        self.init_cond_z_input.grid(row=3, column=1)

        # Input parameters
        self.rec_param_label = tk.Label(self.init_cond_frame, text='Embedding Parameters:')
        self.rec_param_label.grid(row=0, column=2, columnspan=2)
        self.embedding_dim_label_func = tk.Label(self.init_cond_frame, text='m:')
        self.embedding_dim_label_func.grid(row=1, column=2)
        self.time_delay_label_func = tk.Label(self.init_cond_frame, text='T:')
        self.time_delay_label_func.grid(row=2, column=2)
        self.threshold_label_func = tk.Label(self.init_cond_frame, text='E:')
        self.threshold_label_func.grid(row=3, column=2)

        self.embedding_dim_var_func = tk.IntVar(value=1)
        self.time_delay_var_func = tk.IntVar(value=1)
        self.threshold_var_func = tk.IntVar(value=1)

        self.embedding_dim_input_func = tk.Entry(self.init_cond_frame, textvariable=self.embedding_dim_var_func)
        self.embedding_dim_input_func.grid(row=1, column=3)
        self.time_delay_input_func = tk.Entry(self.init_cond_frame, textvariable=self.time_delay_var_func)
        self.time_delay_input_func.grid(row=2, column=3)
        self.threshold_input_func = tk.Entry(self.init_cond_frame, textvariable=self.threshold_var_func)
        self.threshold_input_func.grid(row=3, column=3)

        self.rqa_label_func = ttk.Label(self.command_window_func, text="RQA Measures will appear here")
        self.rqa_label_func.grid(row=0, column=2, rowspan=2)

        self.is_running = False
        self.dt = 0.01
        # self.xyzs = np.array([[self.init_cond_x_var.get(), self.init_cond_y_var.get(), self.init_cond_z_var.get()]])


    def create_home_tab(self):
        self.title_label = ttk.Label(self.home_tab, text="Welcome to Live Data and Recurrence Plot GUI", font=("Helvetica", 16))
        self.title_label.pack(pady=10)

        self.info_text = tk.Text(self.home_tab, wrap='word', height=10, width=50)
        self.info_text.insert(tk.END,
                              "This GUI allows you to visualize different chaotic systems and analyze their recurrence plots. "
                              "You can start/stop/reset the live plotting of selected functions and see the recurrence quantification analysis (RQA) measures.")
        self.info_text.config(state=tk.DISABLED)
        self.info_text.pack(pady=10)

        self.btn_functions_tab = ttk.Button(self.home_tab, text="Go to Plotting Functions", command=lambda: self.notebook.select(self.function_tab))
        self.btn_functions_tab.pack(pady=5)

        self.btn_data_tab = ttk.Button(self.home_tab, text="Go to Plotting Data", command=lambda: self.notebook.select(self.data_tab))
        self.btn_data_tab.pack(pady=5)

    def create_data_tab(self):
        self.data_tab.columnconfigure(0, weight=1)
        self.data_tab.columnconfigure(1, weight=1)
        self.data_tab.rowconfigure(0, weight=1)
        self.data_tab.rowconfigure(1, weight=1)

        self.fig_data = plt.Figure()
        self.fig_rp_data = plt.Figure()

        self.canvas_comp_data = FigureCanvasTkAgg(self.fig_data, master=self.data_tab)
        self.canvas_comp_data.get_tk_widget().grid(row=1, column=0, sticky='nsew')

        self.canvas_rp_data = FigureCanvasTkAgg(self.fig_rp_data, master=self.data_tab)
        self.canvas_rp_data.get_tk_widget().grid(row=1, column=1, sticky='nsew')

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

        self.btn_load_file = ttk.Button(self.command_window_data, text="Load CSV", command=self.load_csv_data)
        self.btn_load_file.grid(row=0, column=0)

        self.btn_run_data = ttk.Button(self.command_window_data, text="Run", command=self.run_data)
        self.btn_run_data.grid(row=1, column=0)

        self.btn_reset_data = ttk.Button(self.command_window_data, text="Reset", command=self.reset_data)
        self.btn_reset_data.grid(row=2, column=0)


        self.embedding_dim_var = tk.IntVar(value=1)
        self.time_delay_var = tk.IntVar(value=1)
        self.threshold_var = tk.DoubleVar(value=1)

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

        self.rqa_label_data = ttk.Label(self.command_window_data, text="RQA Measures will appear here")
        self.rqa_label_data.grid(row=0, column=3, rowspan=2)

    def load_csv_data(self):
        file_path = fd.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.data_np = pd.read_csv(file_path)
            self.data = self.data_np.to_numpy().flatten()  # Flatten to 1D array for simplicity
            self.run_data()
            self.plot_data()

    def update_data_plot(self):
        if self.data.size > 0:
            self.fig_data.clear()
            ax_data = self.fig_data.add_subplot(111)
            ax_data.plot(self.data, lw=0.5)
            ax_data.set_title("Visualization of Inputted Data")
            ax_data.set_xlabel("Index")
            ax_data.set_ylabel("Value")
            self.canvas_comp_data.draw()

    def toggle_buttons(self, state):
        self.btn_reset_func.config(state=state)
        self.dropdown_func.config(state=state)

    def plot_data(self):
        if not self.is_running:
            return

        if self.data.size == 0:
            file_path = 'vibration_data_synthetic.csv'
            self.data_np = pd.read_csv(file_path, sep='\s+')
            self.data = self.data_np.to_numpy().reshape(-1,1)
        self.data_trunc = self.data[:1000]

        m = self.embedding_dim_var.get()
        T = self.time_delay_var.get()
        epsilon = self.threshold_var.get()

        print(m, T, epsilon)

        num_vectors = len(self.data_trunc) - (m - 1) * T
        vectors = np.array([self.data_trunc[t:t + m * T:T] for t in range(num_vectors)]).reshape(-1,1)

        self.D = squareform(pdist(vectors, metric='euclidean'))
        recurrence_matrix = self.D < epsilon
        x_rec, y_rec = np.argwhere(recurrence_matrix).T

        self.fig_data.clear()
        ax_data = self.fig_data.add_subplot(111)
        ax_data.plot(self.data, lw=0.5)
        ax_data.set_title("Visualization of Inputted Data")
        ax_data.set_xlabel("X Axis")
        ax_data.set_ylabel("Y Axis")
        self.canvas_comp_data.draw()

        self.fig_rp_data.clear()
        ax_rp_data = self.fig_rp_data.add_subplot(111)
        ax_rp_data.scatter(x_rec, y_rec, s=1)
        ax_rp_data.set_title("Recurrence Plot")
        ax_rp_data.set_xlabel("Vector Index")
        ax_rp_data.set_ylabel("Vector Index")
        self.canvas_rp_data.draw()

        rqa_measures_data = calculate_rqa_measures_pyrqa(vectors, epsilon)
        det2, lam2, lmax2 = calculate_manual_det_lam_lmax(recurrence_matrix)
        rqa_measures_data["DET2"] = det2
        rqa_measures_data["LAM2"] = lam2
        rqa_measures_data["Lmax2"] = lmax2
        display_rqa_measures(self.rqa_label_data, rqa_measures_data)

    def update_plot(self):
        if not self.is_running:
            return
        self.time_step = self.time_step_var.get()
        print(self.xyzs[0])
        for i in range(self.time_step):
            selected_function_name = self.selected_option_func.get().lower()
            if selected_function_name == 'lorenz':
                new_point = self.xyzs[-1] + lorenz(self.xyzs[-1]) * self.dt
            elif selected_function_name == 'chua':
                new_point = self.xyzs[-1] + chua(self.xyzs[-1]) * self.dt
            elif selected_function_name == 'rossler':
                new_point = self.xyzs[-1] + chua(self.xyzs[-1]) * self.dt
            elif selected_function_name == 'chen':
                new_point = self.xyzs[-1] + chua(self.xyzs[-1]) * self.dt
            else:
                new_point = self.xyzs[-1]

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
            self.D = squareform(pdist(vectors, metric='euclidean'))
            D_max = np.max(self.D)
            epsilon = 0.1 * D_max
            recurrence_matrix = self.D < epsilon
            x_rec, y_rec = np.argwhere(recurrence_matrix).T
            ax_rp_func.scatter(x_rec, y_rec, s=1)
            ax_rp_func.set_title("Recurrence Plot")
            ax_rp_func.set_xlabel("Vector Index")
            ax_rp_func.set_ylabel("Vector Index")
            rqa_measures_func = calculate_rqa_measures_pyrqa(vectors, epsilon)
            det2, lam2, lmax2 = calculate_manual_det_lam_lmax(recurrence_matrix)
            rqa_measures_func["DET2"] = det2
            rqa_measures_func["LAM2"] = lam2
            rqa_measures_func["Lmax2"] = lmax2
            display_rqa_measures(self.rqa_label_func, rqa_measures_func)


    def start_func(self):
        if not self.is_running:
            self.is_running = True
            self.toggle_buttons("disabled")
            self.xyzs = np.array([[self.init_cond_x_var.get(), self.init_cond_y_var.get(), self.init_cond_z_var.get()]])
            self.update_plot()

    def stop_func(self):
        if self.is_running:
            self.is_running = False
            self.toggle_buttons("normal")

    def reset_func(self):
        self.is_running = False
        self.xyzs = np.array([[0., 1., 1.05]])
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
        self.selected_value = self.selected_option_func.get()


if __name__ == "__main__":
    root = tk.Tk()
    app = LivePlotApp(root)
    root.mainloop()