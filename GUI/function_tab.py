import tkinter as tk
import tkinter.filedialog as fd
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.spatial.distance import pdist, squareform
import importlib.util
import matplotlib.pyplot as plt
import numpy as np
import threading

from attractor_functions import *
from rqa_functions import *


class funcTab:
    def __init__(self, root, notebook):
        # access root and notebook from main file
        self.root = root
        self.notebook = notebook

        self.style = ttk.Style()
        self.style.configure('TFrame', backgorund='pink')

        # create function tab
        self.function_tab = ttk.Frame(notebook)
        notebook.add(self.function_tab, text='Plotting Functions')
        self.create_functions_tab()

        # Initialize variables
        self.is_running = False
        self.dt = 0.01
        self.xyzs = np.array([])

    def create_functions_tab(self):
        # set up layout
        self.command_window_func = ttk.Frame(self.function_tab)
        self.command_window_func.place(x=0, y=0, relwidth=0.55, relheight=0.5)

        # set up figures
        self.fig_ps_func = plt.Figure()
        self.fig_comp_func = plt.Figure()
        self.fig_rp_func = plt.Figure()

        self.canvas_ps_func = FigureCanvasTkAgg(self.fig_ps_func, master=self.function_tab)
        self.canvas_ps_func.get_tk_widget().place(relx=0.7, rely=0, relwidth=0.3, relheight=0.5)

        self.canvas_comp_func = FigureCanvasTkAgg(self.fig_comp_func, master=self.function_tab)
        self.canvas_comp_func.get_tk_widget().place(relx=0, rely=0.5, relwidth=0.5, relheight=0.5)

        self.canvas_rp_func = FigureCanvasTkAgg(self.fig_rp_func, master=self.function_tab)
        self.canvas_rp_func.get_tk_widget().place(relx=0.5, rely=0.5, relwidth=0.5, relheight=0.5)

        # set up command window
        self.command_window_func.columnconfigure(0, weight=1)
        self.command_window_func.columnconfigure(1, weight=2)
        self.command_window_func.columnconfigure(2, weight=1)
        # self.command_window_func.rowconfigure(0, weight=1)
        # self.command_window_func.rowconfigure(1, weight=1)

        self.general_controls_frame = ttk.Frame(self.command_window_func)
        self.general_controls_frame.grid(row=0, column=0)

        self.user_input_frame = ttk.Frame(self.command_window_func)
        self.user_input_frame.grid(row=0, column=1)

        self.init_cond_frame = tk.Frame(self.user_input_frame)
        self.init_cond_frame.pack()

        self.init_cond_frame.columnconfigure(0, weight=1)
        self.init_cond_frame.columnconfigure(1, weight=1)
        # self.init_cond_frame.columnconfigure(2, weight=1)
        # self.init_cond_frame.columnconfigure(3, weight=1)
        self.init_cond_frame.rowconfigure(0, weight=1)
        self.init_cond_frame.rowconfigure(1, weight=1)
        self.init_cond_frame.rowconfigure(2, weight=1)
        self.init_cond_frame.rowconfigure(3, weight=1)

        self.embedding_parameter_frame = tk.Frame(self.user_input_frame)
        self.embedding_parameter_frame.pack()

        self.embedding_parameter_frame.columnconfigure(0, weight=1)
        self.embedding_parameter_frame.columnconfigure(1, weight=1)
        # self.init_cond_frame.columnconfigure(2, weight=1)
        # self.init_cond_frame.columnconfigure(3, weight=1)
        self.embedding_parameter_frame.rowconfigure(0, weight=1)
        self.embedding_parameter_frame.rowconfigure(1, weight=1)
        self.embedding_parameter_frame.rowconfigure(2, weight=1)
        self.embedding_parameter_frame.rowconfigure(3, weight=1)

        # configure command window
        # general controls
        self.btn_start_func = ttk.Button(self.general_controls_frame, text="Start", command=self.start_func)
        # self.btn_start_func.grid(row=0, column=0)
        self.btn_start_func.pack(padx=10)

        self.btn_stop_func = ttk.Button(self.general_controls_frame, text="Pause", command=self.stop_func)
        # self.btn_stop_func.grid(row=1, column=0)
        self.btn_stop_func.pack(padx=10)

        self.btn_reset_func = ttk.Button(self.general_controls_frame, text="Reset", command=self.reset_func)
        # self.btn_reset_func.grid(row=2, column=0)
        self.btn_reset_func.pack(padx=10)

        self.selected_option_func = tk.StringVar()
        functions = ["Lorenz", "Chua", "Rossler", "Chen"]
        self.selected_option_func.set(functions[0])
        self.dropdown_func = ttk.OptionMenu(self.general_controls_frame, self.selected_option_func, *functions)
        # self.dropdown_func.grid(row=3, column=0)
        self.dropdown_func.pack(padx=10)
        self.selected_option_func.trace("w", self.on_select)

        self.btn_load_file = ttk.Button(self.general_controls_frame, text="Load Function",
                                        command=self.load_python_module)
        # self.btn_load_file.grid(row=4, column=0)
        self.btn_load_file.pack(padx=10)

        # Inputs for initial conditions
        self.init_cond_label = tk.Label(self.init_cond_frame, text='Initial Conditions:')
        self.init_cond_label.grid(row=0, column=0, columnspan=2)
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

        # Inputs for embedding parameters
        self.rec_param_label = tk.Label(self.embedding_parameter_frame, text='Embedding Parameters:')
        self.rec_param_label.grid(row=0, column=0, columnspan=2)
        self.embedding_dim_label_func = tk.Label(self.embedding_parameter_frame, text='m:')
        self.embedding_dim_label_func.grid(row=1, column=0)
        self.time_delay_label_func = tk.Label(self.embedding_parameter_frame, text='T:')
        self.time_delay_label_func.grid(row=2, column=0)
        self.threshold_label_func = tk.Label(self.embedding_parameter_frame, text='E:')
        self.threshold_label_func.grid(row=3, column=0)

        self.embedding_dim_var_func = tk.IntVar(value=1)
        self.time_delay_var_func = tk.IntVar(value=1)
        self.threshold_var_func = tk.DoubleVar(value=0.1)
        self.threshold_check_var = tk.BooleanVar(value=True)

        self.embedding_dim_input_func = tk.Entry(self.embedding_parameter_frame, textvariable=self.embedding_dim_var_func)
        self.embedding_dim_input_func.grid(row=1, column=1)

        self.time_delay_input_func = tk.Entry(self.embedding_parameter_frame, textvariable=self.time_delay_var_func)
        self.time_delay_input_func.grid(row=2, column=1)

        self.threshold_frame = ttk.Frame(self.embedding_parameter_frame)
        self.threshold_frame.grid(row=3, column=1)
        vcmd = (self.init_cond_frame.register(self.validate_threshold), '%P')
        self.threshold_input_func = ttk.Entry(self.threshold_frame, textvariable=self.threshold_var_func,
                                              validate='focusout', validatecommand=vcmd)
        self.threshold_input_func.pack(side='left')

        self.threshold_check_button = ttk.Checkbutton(self.threshold_frame, text='threshold', variable=self.threshold_check_var)
        self.threshold_check_button.pack(side='left', padx=20)

        # Animation speed Control
        self.slider_frame = ttk.Frame(self.user_input_frame)
        self.slider_frame.pack(pady=20, fill='x')

        self.time_step_var = tk.IntVar(value=1)
        self.slider_label = ttk.Label(self.slider_frame, text="Animation Speed:")
        self.slider_label.pack(side='left')
        self.time_step_slider = ttk.Scale(self.slider_frame, from_=1, to=100, variable=self.time_step_var,
                                          orient=tk.HORIZONTAL)
        self.time_step_slider.pack(fill='x', padx=20)

        # Select time series coordinate
        self.coordinate_frame = ttk.Frame(self.user_input_frame)
        self.coordinate_frame.pack(pady=20, fill='x')

        self.coord_label = ttk.Label(self.coordinate_frame, text="Variable of Interest:")
        self.coord_label.pack(side='left')

        self.check_var_x = tk.BooleanVar(value=True)
        self.check_var_y = tk.BooleanVar()
        self.check_var_z = tk.BooleanVar()
        self.check_button_x = ttk.Checkbutton(self.coordinate_frame, text='x',
                                              command=self.reset_func, variable=self.check_var_x)
        self.check_button_x.pack(side='left', padx=10)
        self.check_button_y = ttk.Checkbutton(self.coordinate_frame, text='y',
                                              command=self.reset_func, variable=self.check_var_y)
        self.check_button_y.pack(side='left', padx=10)
        self.check_button_z = ttk.Checkbutton(self.coordinate_frame, text='z',
                                              command=self.reset_func, variable=self.check_var_z)
        self.check_button_z.pack(side='left', padx=10)

        # RQA display
        self.rqa_label_func = ttk.Label(self.command_window_func, text="RQA Measures will appear here")
        self.rqa_label_func.grid(row=0, column=2, rowspan=2)

    def validate_threshold(self, value_if_allowed):
        if not value_if_allowed:
            self.threshold_var_func.set(0.1)
            return True
        try:
            value = float(value_if_allowed)
            if 0.0 <= value <= 1.0:
                return True
        except ValueError:
            pass
        self.show_error("Please enter a float between 0.0 and 1.0.")
        self.threshold_var_func.set(0.1)
        return False

    def show_error(self, message):
        messagebox.showerror("Input Error", message)

    def load_python_module(self):
        file_path = fd.askopenfilename(filetypes=[("Python files", "*.py")])
        if file_path:
            module_name = file_path.split("/")[-1].split(".")[0]
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            # spec.loader.exec_module(module)
            # self.loaded_module = module
            print(f"Module {module_name} loaded.")

    def toggle_buttons(self, state):
        buttons = [self.btn_reset_func, self.dropdown_func, self.embedding_dim_input_func, self.time_delay_input_func,
                   self.threshold_input_func, self.init_cond_x_input, self.init_cond_y_input, self.init_cond_z_input]
        for button in buttons:
            button.configure(state=state)

    def update_plot(self):
        if not self.is_running:
            return

        # time step determines speed of animation, controlled by scroller
        self.time_step = self.time_step_var.get()
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

        # plot phase space of function
        self.fig_ps_func.clear()
        ax_ps = self.fig_ps_func.add_subplot(111, projection='3d')
        ax_ps.plot(*self.xyzs.T, lw=0.5)
        ax_ps.set_title(f"{selected_function_name.capitalize()} Attractor")
        ax_ps.set_xlabel("X Axis")
        ax_ps.set_ylabel("Y Axis")
        ax_ps.set_zlabel("Z Axis")

        # plot component of function which acts as time series
        # check which coordinates are selected
        self.coord_active = [self.check_var_x.get(), self.check_var_y.get(), self.check_var_z.get()]
        coord_key = ['X', 'Y', 'Z']
        if sum(self.coord_active) == 0:  # No plots
            self.show_error('please select at least one coordinate to create the recurrence plot')

        elif sum(self.coord_active) == 1:  # univariate recurrence plot
            # reactivate ability to unthreshold
            self.threshold_check_button.configure(state='normal')

            # extract activated time series
            true_index = self.coord_active.index(True)
            self.x = [coord[true_index] for coord in self.xyzs]  # introduce functionality to select which time series

            # plot time series of activated component
            self.fig_comp_func.clear()
            ax_comp = self.fig_comp_func.add_subplot(111)
            ax_comp.plot(self.x, lw=0.5)
            ax_comp.set_title(f"{coord_key[true_index]}-coordinate")
            ax_comp.set_xlabel("Time")
            ax_comp.set_ylabel(f"{coord_key[true_index]}-coordinate")

            # calculates recurrence plot
            recurrence_matrix = self.calculate_recurrence_plot(self.x)
            self.calculate_rqa(recurrence_matrix)
            self.update_recurrence_plot_figure(recurrence_matrix, 'Recurrence')

        elif sum(self.coord_active) > 1:  # multivariate joint recurrence plot
            # deactive ability to unthreshold
            self.threshold_check_var.set(True)
            self.threshold_check_button.configure(state='disabled')

            # actualize joint recurrence plot array
            jrp = np.zeros_like(self.xyzs[:,0])

            # set up plot
            self.fig_comp_func.clear()
            ax_comp = self.fig_comp_func.add_subplot(111)
            ax_comp.set_title("Individual components")
            ax_comp.set_xlabel("Time")
            ax_comp.set_ylabel("Value")

            # iterate through activated components
            for i in range(len(self.coord_active)):
                if self.coord_active[i]:
                    # get time series
                    data = [coord[i] for coord in self.xyzs]

                    # add time series to plot
                    ax_comp.plot(data, lw=0.5, label=f'{coord_key[i]}-coordinate')
                    ax_comp.legend()

                    rp = self.calculate_recurrence_plot(data)
                    if np.all(jrp == 0):  # for the first rp, set jrp to rp
                        jrp = rp
                    else:  # afterward check with and gate to create jrp
                        jrp = np.logical_and(jrp, rp)
            self.update_recurrence_plot_figure(jrp, 'Joint Recurrence')

        # display 3 figures
        self.canvas_ps_func.draw()
        self.canvas_comp_func.draw()
        self.canvas_rp_func.draw()

        # continue updating while is_running
        if self.is_running:
            self.root.after(100, self.update_plot)

    def calculate_recurrence_plot(self, data):
        # get embedding parameters from user input
        self.m = self.embedding_dim_var_func.get()
        self.T = self.time_delay_var_func.get()
        self.epsilon = self.threshold_var_func.get()

        # embed time series
        self.num_vectors = len(data) - (self.m - 1) * self.T
        self.vectors = np.array([data[t:t + self.m * self.T:self.T] for t in range(self.num_vectors)])

        if self.vectors.size > 0:  # check that enough points exist to create recurrence plot
            # create and normalize similarity matrix
            self.D = squareform(pdist(self.vectors, metric='euclidean'))
            D_max = np.max(self.D)
            self.D_norm = self.D / D_max

            # create recurrence matrix
            if self.threshold_check_var.get():
                recurrence_matrix = self.D_norm < self.epsilon
            else:
                recurrence_matrix = self.D_norm
            return recurrence_matrix

    def update_recurrence_plot_figure(self, recurrence_matrix, type):
        # set up figure
        self.fig_rp_func.clear()
        ax_rp_func = self.fig_rp_func.add_subplot(111)

        # plot recurrence matrix
        # ax_rp_func.scatter(x_rec, y_rec, s=1)
        im = ax_rp_func.imshow(recurrence_matrix, cmap='binary', origin='lower')
        # self.fig_rp_func.colorbar(im, ax=ax_rp_func)
        ax_rp_func.set_title(f"{type} Plot (embedding dim: {self.m}; time delay: {self.T}) {self.threshold_check_var.get()}")
        ax_rp_func.set_xlabel("Vector Index")
        ax_rp_func.set_ylabel("Vector Index")

    def calculate_rqa(self, recurrence_matrix):
        # calculate and display RQAs
        rqa_measures_func = calculate_rqa_measures_pyrqa(self.vectors, self.epsilon)
        det2, lam2, lmax2 = calculate_manual_det_lam_lmax(recurrence_matrix)
        rqa_measures_func["DET2"] = det2
        rqa_measures_func["LAM2"] = lam2
        rqa_measures_func["Lmax2"] = lmax2
        display_rqa_measures(self.rqa_label_func, rqa_measures_func)

    def start_func(self):
        if not self.is_running:
            self.is_running = True
            self.toggle_buttons("disabled")  # deactivates all user inputs

            # set initial conditions if time series is empty, otherwise do nothing and plotting will continue from pause
            if len(self.xyzs) == 0:
                self.xyzs = np.array(
                    [[self.init_cond_x_var.get(), self.init_cond_y_var.get(), self.init_cond_z_var.get()]])

            # start plotting
            self.update_plot()

    def stop_func(self):
        if self.is_running:
            self.is_running = False
            self.toggle_buttons("normal")  # reactivates all user inputs

    def reset_func(self):
        self.is_running = False
        self.xyzs = np.array([])
        self.fig_ps_func.clear()
        self.fig_comp_func.clear()
        self.fig_rp_func.clear()
        self.canvas_ps_func.draw()
        self.canvas_comp_func.draw()
        self.canvas_rp_func.draw()

    def on_select(self, *args):
        self.selected_value = self.selected_option_func.get()
