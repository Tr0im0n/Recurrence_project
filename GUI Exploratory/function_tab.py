import tkinter as tk
import tkinter.filedialog as fd
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.spatial.distance import pdist, squareform
import importlib.util
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np

from attractor_functions import *
from rqa_functions import *


class funcTab:
    def __init__(self, root, notebook):
        # access root and notebook from main file
        self.root = root
        self.notebook = notebook

        # set styling
        self.style = ttk.Style()
        self.style.configure('TFrame', background='white')

        # Initialize variables
        self.is_running = False
        self.dt = 0.01
        self.xyzs = np.array([])

        self.rqa_measures = {
            "RR": 'TBD',
            "DET": 'TBD',
            "L": 'TBD',
            "Lmax": 'TBD',
            "DIV": 'TBD',
            "ENTR": 'TBD',
            "LAM": 'TBD',
            "TT": 'TBD'
        }

        self.start_index = 0  # keeps track of points past 1000

        # create function tab
        self.function_tab = tk.Frame(notebook)
        notebook.add(self.function_tab, text='Plotting Functions')
        self.create_functions_tab()

    def embedding_param_layout(self):
        self.embedding_param_frame = ttk.Frame(self.command_window_frame_top)
        self.embedding_param_frame.pack(side='left', padx=20, anchor='c', expand=True, fill='x')

        self.embedding_param_frame.columnconfigure(0,weight=1)
        self.embedding_param_frame.columnconfigure(1, weight=3)
        self.embedding_param_frame.columnconfigure(2, weight=1)

        # Inputs for embedding parameters
        self.embedding_param_label = ttk.Label(self.embedding_param_frame, text='Embedding Parameters')
        self.embedding_param_label.grid(row=0, column=0, columnspan=3)
        self.embedding_dim_label = ttk.Label(self.embedding_param_frame, text='m:')
        self.embedding_dim_label.grid(row=1, column=0, padx=(10, 0), pady=(0, 10))
        self.time_delay_label = ttk.Label(self.embedding_param_frame, text='T:')
        self.time_delay_label.grid(row=2, column=0, padx=(10, 0))
        self.threshold_label = ttk.Label(self.embedding_param_frame, text='E:')
        self.threshold_label.grid(row=3, column=0, padx=(10, 0), pady=10)

        self.embedding_dim_var = tk.IntVar(value=1)
        self.time_delay_var = tk.IntVar(value=1)
        self.threshold_var = tk.DoubleVar(value=0.1)
        self.threshold_check_var = tk.BooleanVar(value=True)

        vcmd_param = (self.embedding_param_frame.register(self.validate_parameter), '%P')
        self.embedding_dim_input = ttk.Entry(self.embedding_param_frame, textvariable=self.embedding_dim_var, width=5,
                                             validate='focusout', validatecommand=vcmd_param)
        self.embedding_dim_input.grid(row=1, column=1, columnspan=2, sticky='ew', padx=10, pady=(0, 10))

        self.time_delay_input = ttk.Entry(self.embedding_param_frame, textvariable=self.time_delay_var, width=5,
                                          validate='focusout', validatecommand=vcmd_param)
        self.time_delay_input.grid(row=2, column=1, columnspan=2, sticky='ew', padx=10)

        vcmd_thresh = (self.embedding_param_frame.register(self.validate_threshold), '%P')
        self.threshold_input = ttk.Entry(self.embedding_param_frame, textvariable=self.threshold_var, width=5,
                                         validate='focusout', validatecommand=vcmd_thresh)
        self.threshold_input.grid(row=3, column=1, sticky='w', padx=(10, 0), pady=10)

        self.threshold_check_button = ttk.Checkbutton(self.embedding_param_frame, text='threshold', variable=self.threshold_check_var)
        self.threshold_check_button.grid(row=3,column=2, sticky='e', padx=(0,10), pady=10)

    def init_cond_layout(self):
        self.init_cond_frame = ttk.Frame(self.command_window_frame_top)
        self.init_cond_frame.pack(side='left', padx=(0,20), anchor='c', expand=True, fill='x')

        self.init_cond_frame.columnconfigure(0, weight=1)
        self.init_cond_frame.columnconfigure(1, weight=8)

        # Inputs for initial conditions
        self.init_cond_label = ttk.Label(self.init_cond_frame, text='Initial Conditions')
        self.init_cond_label.grid(row=0, column=0, columnspan=2)
        self.init_cond_x_label = ttk.Label(self.init_cond_frame, text='X:')
        self.init_cond_x_label.grid(row=1, column=0, padx=(10, 0), pady=(0, 10))
        self.init_cond_y_label = ttk.Label(self.init_cond_frame, text='Y:')
        self.init_cond_y_label.grid(row=2, column=0, padx=(10, 0), pady=(0, 10))
        self.init_cond_z_label = ttk.Label(self.init_cond_frame, text='Z:')
        self.init_cond_z_label.grid(row=3, column=0, padx=(10, 0), pady=(0, 10))

        self.init_cond_x_var = tk.DoubleVar(value=1)
        self.init_cond_y_var = tk.DoubleVar(value=1)
        self.init_cond_z_var = tk.DoubleVar(value=1)

        self.update_previous_values()

        vcmd_init_cond = (self.embedding_param_frame.register(self.validate_init_cond), '%P')

        self.init_cond_x_input = ttk.Entry(self.init_cond_frame, textvariable=self.init_cond_x_var, width=10,
                                           validate='focusout', validatecommand=vcmd_init_cond)
        self.init_cond_x_input.grid(row=1, column=1, sticky='ew', padx=(10, 0), pady=(0, 10))
        self.init_cond_y_input = ttk.Entry(self.init_cond_frame, textvariable=self.init_cond_y_var, width=10,
                                           validate='focusout', validatecommand=vcmd_init_cond)
        self.init_cond_y_input.grid(row=2, column=1, sticky='ew', padx=(10, 0), pady=(0, 10))
        self.init_cond_z_input = ttk.Entry(self.init_cond_frame, textvariable=self.init_cond_z_var, width=10,
                                           validate='focusout', validatecommand=vcmd_init_cond)
        self.init_cond_z_input.grid(row=3, column=1, sticky='ew', padx=(10, 0), pady=(0, 10))

    def general_controls_layout(self):
        self.general_controls_frame = ttk.Frame(self.command_window_frame_bottom)
        self.general_controls_frame.pack(anchor='c', expand=True, fill='y', pady=(0,20))

        self.btn_start = ttk.Button(self.general_controls_frame, text="Start", command=self.start_func)
        self.btn_start.pack(side='left', padx=10, fill='x', anchor='s')

        self.btn_stop = ttk.Button(self.general_controls_frame, text="Pause", command=self.stop_func)
        self.btn_stop.pack(side='left', padx=10, fill='x', anchor='s')

        self.btn_reset = ttk.Button(self.general_controls_frame, text="Reset", command=self.reset_func)
        self.btn_reset.pack(side='left', padx=10, fill='x', anchor='s')

    def function_selection_layout(self):
        self.function_selection_frame = ttk.Frame(self.command_window_frame_middle)
        self.function_selection_frame.pack(side='left',pady=(20), padx=(30,0))

        self.label_option = ttk.Label(self.function_selection_frame, text='Select Function: ')
        self.label_option.pack(padx=10)

        self.selected_option_func = tk.StringVar()
        functions = ["Lorenz", "Chua", "Rossler", "Chen", "upload function"]
        self.selected_option_func.set(functions[0])
        self.dropdown_func = ttk.OptionMenu(self.function_selection_frame, self.selected_option_func, functions[0], *functions)
        self.dropdown_func.pack(padx=10, fill='x')
        self.selected_option_func.trace("w", self.on_select)

        # initialize variables
        self.check_function_uploaded = False

        # self.upload_text = tk.StringVar(value='Load Function')
        # self.btn_load_file = ttk.Button(self.function_selection_frame, textvariable=self.upload_text,
        #                                 command=self.load_python_module)
        # self.btn_load_file.pack(padx=10)

    def speed_control_layout(self):
        self.slider_frame = ttk.Frame(self.command_window_frame_middle)
        self.slider_frame.pack(anchor='e', padx=5, pady=20)

        self.time_step_var = tk.IntVar(value=1)
        self.slider_label = ttk.Label(self.slider_frame, text="Animation Speed:")
        self.slider_label.pack(side='left', padx=(10,0))
        self.time_step_slider = ttk.Scale(self.slider_frame, from_=1, to=100, variable=self.time_step_var,
                                          orient=tk.HORIZONTAL, length=200)
        self.time_step_slider.pack(fill='x', padx=20)

    def select_timeseries_layout(self):
        self.coordinate_frame = ttk.Frame(self.command_window_frame_middle)
        self.coordinate_frame.pack(anchor='e', padx=(5,25))

        self.coord_label = ttk.Label(self.coordinate_frame, text="Variable of Interest:")
        self.coord_label.pack(side='left', padx=(10,0))

        self.coordinate_box_frame = ttk.Frame(self.coordinate_frame, width=200)
        self.coordinate_box_frame.pack(anchor='e', padx=(20,0))

        self.check_var_x = tk.BooleanVar(value=True)
        self.check_var_y = tk.BooleanVar()
        self.check_var_z = tk.BooleanVar()
        self.check_var_CRP = tk.BooleanVar()
        self.check_button_x = ttk.Checkbutton(self.coordinate_box_frame, text='x',
                                              command=self.reset_func, variable=self.check_var_x)
        self.check_button_x.pack(side='left', padx=(0,10))
        self.check_button_y = ttk.Checkbutton(self.coordinate_box_frame, text='y',
                                              command=self.reset_func, variable=self.check_var_y)
        self.check_button_y.pack(side='left', padx=10)
        self.check_button_z = ttk.Checkbutton(self.coordinate_box_frame, text='z',
                                              command=self.reset_func, variable=self.check_var_z)
        self.check_button_z.pack(side='left', padx=10)
        self.check_button_CRP = ttk.Checkbutton(self.coordinate_box_frame, text='CRP',
                                              command=self.reset_func, variable=self.check_var_CRP)
        self.check_button_CRP.pack(side='left', padx=(10,0))

    def display_rqa_measures(self):
        self.rqa_frame = ttk.Frame(self.function_tab)
        self.rqa_frame.place(relx=0.4, rely=0, relheight=0.45, relwidth=0.16)

        self.rqa_table = tk.Frame(self.rqa_frame)
        self.rqa_table.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # create the treeview
        self.tree = ttk.Treeview(self.rqa_table, columns=("Measure", "Value"), show='headings', height=10)
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.tree.heading("Measure", text="Measure")
        self.tree.heading("Value", text="Value")

        for measure, value in self.rqa_measures.items():
            self.tree.insert("", "end", values=(measure, value))

        # Configure the column width
        self.tree.column("Measure", anchor=tk.CENTER, width=100)
        self.tree.column("Value", anchor=tk.CENTER, width=100)


    def create_functions_tab(self):
        # set up figures
        self.ps_plot_frame = ttk.Frame(self.function_tab)
        self.ps_plot_frame.place(relx=0.5, rely=0, relwidth=0.5, relheight=0.45)

        self.fig_ps_func = plt.Figure()
        self.fig_comp_func = plt.Figure()
        self.fig_rp_func = plt.Figure()

        self.canvas_ps_func = FigureCanvasTkAgg(self.fig_ps_func, master=self.ps_plot_frame)
        self.canvas_ps_func.get_tk_widget().pack(fill='both', expand=True)

        self.canvas_comp_func = FigureCanvasTkAgg(self.fig_comp_func, master=self.function_tab)
        self.canvas_comp_func.get_tk_widget().place(relx=0, rely=0.45, relwidth=0.5, relheight=0.55)

        self.canvas_rp_func = FigureCanvasTkAgg(self.fig_rp_func, master=self.function_tab)
        self.canvas_rp_func.get_tk_widget().place(relx=0.5, rely=0.45, relwidth=0.5, relheight=0.55)

        # set up command window
        self.command_window_frame_top = ttk.Frame(self.function_tab)
        self.command_window_frame_top.place(relx=0, rely=0, relwidth=0.4, relheight=0.2)

        self.separator = ttk.Separator(self.function_tab, orient='horizontal')
        self.separator.place(relx=0, rely=0.2)

        self.command_window_frame_middle = ttk.Frame(self.function_tab)
        self.command_window_frame_middle.place(relx=0, rely=0.2, relwidth=0.4, relheight=0.15)

        self.command_window_frame_bottom = ttk.Frame(self.function_tab)
        self.command_window_frame_bottom.place(relx=0, rely=0.35, relwidth=0.4, relheight=0.1)

        # set up control panel
        self.general_controls_layout()
        self.embedding_param_layout()
        self.init_cond_layout()
        self.function_selection_layout()
        self.speed_control_layout()
        self.select_timeseries_layout()
        self.display_rqa_measures()

    def validate_threshold(self, value_if_allowed):
        if not value_if_allowed:
            self.revert_to_previous_values()
            return True
        try:
            value = float(value_if_allowed)
            if 0.0 <= value <= 1.0:
                self.update_previous_values()
                return True
        except ValueError:
            pass
        self.show_error("Please enter a float between 0.0 and 1.0.")
        self.revert_to_previous_values()
        return False

    def validate_parameter(self, value_if_allowed):
        if not value_if_allowed:
            self.revert_to_previous_values()
            return True
        try:
            value = int(value_if_allowed)
            if 0 < value:
                self.update_previous_values()
                return True
        except ValueError:
            pass
        self.show_error("Please enter a positive integer")
        self.revert_to_previous_values()
        return False

    def validate_init_cond(self, value_if_allowed):
        if not value_if_allowed:
            self.revert_to_previous_values()
            return True
        try:
            value = float(value_if_allowed)
            self.update_previous_values()
            return True
        except ValueError:
            pass
        self.show_error("Please enter a real number")
        self.revert_to_previous_values()
        return False

    def show_error(self, message):
        messagebox.showerror("Input Error", message)

    def update_previous_values(self):
        self.previous_values = {
            'init_cond_x': self.init_cond_x_var.get(),
            'init_cond_y': self.init_cond_y_var.get(),
            'init_cond_z': self.init_cond_z_var.get(),
            'embedding_dim': self.embedding_dim_var.get(),
            'time_delay': self.time_delay_var.get(),
            'threshold': self.threshold_var.get()
        }

    def revert_to_previous_values(self):
        self.init_cond_x_var.set(self.previous_values['init_cond_x'])
        self.init_cond_y_var.set(self.previous_values['init_cond_y'])
        self.init_cond_z_var.set(self.previous_values['init_cond_z'])
        self.embedding_dim_var.set(self.previous_values['embedding_dim'])
        self.time_delay_var.set(self.previous_values['time_delay'])
        self.threshold_var.set(self.previous_values['threshold'])

    def load_python_module(self):
        file_path = fd.askopenfilename(filetypes=[("Python files", "*.py")])
        if file_path:
            # Upload the python file
            module_name = file_path.split("/")[-1].split(".")[0]
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)

            # check that file contains function named 'function'
            if hasattr(self.module, 'function'):
                self.check_function_uploaded = True


    def toggle_buttons(self, state):
        buttons = [self.btn_reset, self.dropdown_func, self.embedding_dim_input, self.time_delay_input,
                   self.threshold_input, self.init_cond_x_input, self.init_cond_y_input, self.init_cond_z_input]
        for button in buttons:
            button.configure(state=state)

    def update_plot(self):
        if not self.is_running:
            return

        # time step determines speed of animation, controlled by scroller
        self.time_step = self.time_step_var.get()
        for i in range(self.time_step):
            selected_function_name = self.selected_option_func.get().lower()
            if self.check_function_uploaded:
                new_point = self.xyzs[-1] + self.module.function(self.xyzs[-1]) * self.dt
            elif selected_function_name == 'lorenz':
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

            # drop first point if length > 1000
            if len(self.xyzs) > 1000:
                self.xyzs = self.xyzs[1:]
                self.start_index += 1

        # plot phase space of function
        self.fig_ps_func.clear()
        ax_ps = self.fig_ps_func.add_subplot(111, projection='3d')
        ax_ps.plot(*self.xyzs.T, lw=0.5)
        ax_ps.set_title(f"{selected_function_name.capitalize()} Attractor")
        ax_ps.set_xlabel("X Axis")
        ax_ps.set_ylabel("Y Axis")
        ax_ps.set_zlabel("Z Axis")
        # self.fig_ps_func.tight_layout()

        # plot component of function which acts as time series
        # check which coordinates are selected
        self.coord_active = [self.check_var_x.get(), self.check_var_y.get(), self.check_var_z.get()]
        coord_key = ['X', 'Y', 'Z']
        if sum(self.coord_active) == 0:  # No plots
            self.show_error('please select at least one coordinate to create the recurrence plot')
            self.check_var_x.set(True)
            self.reset_func()
        elif sum(self.coord_active) == 1:  # univariate recurrence plot
            # reactivate ability to unthreshold
            self.threshold_check_button.configure(state='normal')

            # extract activated time series
            true_index = self.coord_active.index(True)
            self.x = [coord[true_index] for coord in self.xyzs]  # introduce functionality to select which time series

            # plot time series of activated component
            self.fig_comp_func.clear()
            ax_comp = self.fig_comp_func.add_subplot(111)
            self.x_values = np.arange(self.start_index, self.start_index + len(self.xyzs))
            ax_comp.plot(self.x_values, self.x, lw=0.5)
            ax_comp.set_title(f"{coord_key[true_index]}-coordinate")
            ax_comp.set_xlabel("Time")
            ax_comp.set_ylabel(f"{coord_key[true_index]}-coordinate")

            # calculates recurrence plot
            recurrence_matrix = self.calculate_recurrence_plot(self.x)
            self.calculate_rqa(recurrence_matrix)
            self.update_recurrence_plot_figure(recurrence_matrix, 'Recurrence')

        elif sum(self.coord_active) == 2 and self.check_var_CRP.get():  # multivariate joint recurrence plot
            # reactivate ability to unthreshold
            self.threshold_check_button.configure(state='normal')

            # set up plot
            self.fig_comp_func.clear()
            ax_comp = self.fig_comp_func.add_subplot(111)
            ax_comp.set_title("Individual components")
            ax_comp.set_xlabel("Time")
            ax_comp.set_ylabel("Value")

            # iterate through activated components
            first = True
            for i in range(len(self.coord_active)):
                if self.coord_active[i]:
                    # get time series
                    data = [coord[i] for coord in self.xyzs]

                    # add time series to plot
                    ax_comp.plot(data, lw=0.5, label=f'{coord_key[i]}-coordinate')
                    ax_comp.legend()

                    if first == True:
                        data1 = data
                        first = False
                    else:
                        data2 = data
                        rp = self.calculate_cross_recurrence_plot(data1, data)

            self.calculate_rqa(rp)
            self.update_recurrence_plot_figure(rp, 'Cross Recurrence')


        elif sum(self.coord_active) > 1:
            # deactivate ability to unthreshold
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
            self.calculate_rqa(jrp)
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
        self.m = self.embedding_dim_var.get()
        self.T = self.time_delay_var.get()
        self.epsilon = self.threshold_var.get()

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
    def calculate_cross_recurrence_plot(self, data1, data2):
        # get embedding parameters from user input
        self.m = self.embedding_dim_var.get()
        self.T = self.time_delay_var.get()
        self.epsilon = self.threshold_var.get()

        self.num_vectors = len(data1) - (self.m - 1) * self.T

        H_ts1 = np.array([data1[i:i + self.m * self.T:self.T] for i in range(self.num_vectors)]).reshape(-1,1)
        H_ts2 = np.array([data2[i:i + self.m * self.T:self.T] for i in range(self.num_vectors)]).reshape(-1,1)

        # Calculate pairwise distances between trajectory matrices
        self.D = cdist(H_ts1, H_ts2, 'euclidean')
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

        # keep track of true index after 1000
        extent = [self.start_index, self.start_index + recurrence_matrix.shape[1],
                  self.start_index, self.start_index + recurrence_matrix.shape[0]]

        # plot recurrence matrix
        im = ax_rp_func.imshow(recurrence_matrix, cmap='binary', origin='lower', extent=extent)
        self.fig_rp_func.colorbar(im, ax=ax_rp_func)
        # ax_rp_func.set_title(f"{type} Plot")
        ax_rp_func.set_xlabel("Vector Index")
        ax_rp_func.set_ylabel("Vector Index")
        self.fig_rp_func.savefig('cross_recurrence_plot.png', dpi=500)

    def calculate_rqa(self, recurrence_matrix):
        self.rqa_measures = calculate_rqa_measures_pyrqa(self.vectors, self.m, self.T, self.epsilon)
        self.display_rqa_measures()

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
        self.start_idnex = 0
        self.rqa_measures = {
            "RR": 'TBD',
            "DET": 'TBD',
            "L": 'TBD',
            "Lmax": 'TBD',
            "DIV": 'TBD',
            "ENTR": 'TBD',
            "LAM": 'TBD',
            "TT": 'TBD'
        }
        self.display_rqa_measures()
        if sum([self.check_var_x.get(),self.check_var_y.get(),self.check_var_z.get()]) == 2:
            self.check_button_CRP.configure(state='normal')
        else:
            self.check_button_CRP.configure(state='disabled')
            self.check_var_CRP.set(False)

    def on_select(self, *args):
        self.check_function_uploaded = False
        self.reset_func()
        self.selected_value = self.selected_option_func.get()
        if self.selected_value == 'upload function':
            self.load_python_module()
