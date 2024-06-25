import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.computation import RQAComputation
from pyrqa.metric import EuclideanMetric
from pyrqa.neighbourhood import FixedRadius
from scipy.spatial.distance import pdist, squareform
from data_feed import make_window
from time import time


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter.filedialog as fd

class dispTab:
    def __init__(self, root, notebook):
        # access root and notebook from main file
        self.root = root
        self.notebook = notebook

        # Initialize variables
        self.is_running = True
        self.path = '/Users/martina/Documents/GitHub/Recurrence_project/Classifier/data/normal_3hp_1730rpm.csv'
        self.start_time = time()
        self.RR = []
        self.DET = []
        self.L = []
        self.Lmax=[]
        self.DIV = []
        self.ENTR = []
        self.LAM = []
        self.TT = []

        # create display tab
        self.disp_tab = tk.Frame(notebook, background='white')
        notebook.add(self.disp_tab, text='Live Window')
        self.create_disp_tab()

    def create_disp_tab(self):
        self.left_frame = tk.Frame(self.disp_tab, background='white')
        self.left_frame.place(relx=0, rely=0, relwidth=0.5, relheight=1)

        self.right_frame = tk.Frame(self.disp_tab, background='white')
        self.right_frame.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)

        self.root.after(2000)
        self.left_window_layout()
        self.right_window_layout()

    def left_window_layout(self):
        # set up live plot figure
        self.live_plot_frame = tk.Frame(self.left_frame, background='white')
        self.live_plot_frame.place(relx=0, rely=0, relwidth=1, relheight=0.5)

        self.fig_live_data = plt.Figure()
        self.canvas_data = FigureCanvasTkAgg(self.fig_live_data, master=self.live_plot_frame)
        self.canvas_data.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

        self.update_data_plot()

        # set up maintenance warning display
        self.warning_frame = tk.Frame(self.left_frame, background='white', bd=2, relief='solid', highlightbackground="black", highlightcolor="black",
                         highlightthickness=1)
        self.warning_frame.place(relx=0.1, rely=0.55, relwidth=0.8, relheight=0.4)

        self.update_warning_data()


    def update_data_plot(self):
        if not self.is_running:
            return

        # determine elapsed time to extract relevant data window
        current_time = time()
        elapsed_time = round(current_time - self.start_time, 1)

        # get data window
        window_size = 1000
        self.data = make_window(self.path, elapsed_time, window_size)

        t = np.linspace(0,window_size, 1000)

        # plot phase space of function
        self.fig_live_data.clear()
        ax_ps = self.fig_live_data.add_subplot(111)
        ax_ps.plot(t, self.data, lw=0.5)
        ax_ps.set_title(f"Live data")
        ax_ps.set_xlabel("Time")
        ax_ps.set_ylabel("Data")
        ax_ps.set_ylim(-1, 1)
        self.fig_live_data.tight_layout()

        self.canvas_data.draw()

        # if self.is_running:
        #     self.root.after(100, self.update_data_plot)

    def update_warning_data(self):
        self.red_light = tk.Radiobutton(self.warning_frame, indicatoron=1, width=3, height=3, bg="grey",
                                        bd=2, highlightbackground="black", highlightcolor="black")
        self.red_light.pack(side="left", padx=20, pady=20)

        self.green_light = tk.Radiobutton(self.warning_frame, indicatoron=0, width=3, height=3, bg="grey",
                                          bd=2, highlightbackground="black", highlightcolor="black")
        self.green_light.pack(side="left", padx=20, pady=20)

        self.toggle_button = ttk.Button(self.warning_frame, text="Toggle Lights", command=self.toggle_lights)
        self.toggle_button.pack(side="left", padx=20, pady=20)

        self.is_red_on = False
        self.is_green_on = False

    def toggle_lights(self):
        if self.is_red_on:
            self.red_light.config(bg="grey")
        else:
            self.red_light.config(bg="red")

        if self.is_green_on:
            self.green_light.config(bg="grey")
        else:
            self.green_light.config(bg="green")

        self.is_red_on = not self.is_red_on
        self.is_green_on = not self.is_green_on

    def right_window_layout(self):
        # set up recurrence plot figure
        self.rp_frame = tk.Frame(self.right_frame, background='white')
        self.rp_frame.place(relx=0, rely=0, relwidth=1, relheight=0.5)

        self.fig_rp = plt.Figure()
        self.canvas_rp = FigureCanvasTkAgg(self.fig_rp, master=self.rp_frame)
        self.canvas_rp.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

        self.update_rp()

        # set up RQA feature plot
        self.calculate_rqa_measures_pyrqa()
        print(self.RR, self.DET)

        self.rqa_frame = tk.Frame(self.right_frame, background='white')
        self.rqa_frame.place(relx=0, rely=0.5, relwidth=1, relheight=0.5)

        self.fig_rqa = plt.Figure()
        self.canvas_rqa = FigureCanvasTkAgg(self.fig_rqa, master=self.rqa_frame)
        self.canvas_rqa.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

        self.update_rqa_plot()

    def update_rp(self):
        # calculates recurrence plot
        recurrence_matrix = self.calculate_recurrence_plot(self.data)
        # self.calculate_rqa(recurrence_matrix)
        # print(self.rqa_measures)

        # set up figure
        self.fig_rp.clear()
        ax_rp_func = self.fig_rp.add_subplot(111)

        # plot recurrence matrix
        im = ax_rp_func.imshow(recurrence_matrix, cmap='binary', origin='lower')
        self.fig_rp.colorbar(im, ax=ax_rp_func)
        ax_rp_func.set_title(f"Recurrence Plot")
        ax_rp_func.set_xlabel("Vector Index")
        ax_rp_func.set_ylabel("Vector Index")

        # display rp
        self.canvas_rp.draw()

    def calculate_recurrence_plot(self, data):
        # get embedding parameters from user input
        # self.m = self.embedding_dim_var.get()
        # self.T = self.time_delay_var.get()
        # self.epsilon = self.threshold_var.get()

        self.m = 1
        self.T = 1
        self.epsilon = 0.1

        # embed time series
        self.num_vectors = len(data) - (self.m - 1) * self.T
        self.vectors = np.array([data[t:t + self.m * self.T:self.T] for t in range(self.num_vectors)])

        if self.vectors.size > 0:  # check that enough points exist to create recurrence plot
            # create and normalize similarity matrix
            self.D = squareform(pdist(self.vectors, metric='euclidean'))
            D_max = np.max(self.D)
            self.D_norm = self.D / D_max

            # create recurrence matrix
            recurrence_matrix = self.D_norm < self.epsilon
            return recurrence_matrix

    def calculate_rqa_measures_pyrqa(self):
        time_series = TimeSeries(self.vectors[:, 0], embedding_dimension=self.m, time_delay=self.T)
        settings = Settings(
            time_series,
            neighbourhood=FixedRadius(self.epsilon),
            similarity_measure=EuclideanMetric(),
            theiler_corrector=1
        )

        computation = RQAComputation.create(settings)
        result = computation.run()

        self.rqa_measures = {
            "RR": round(result.recurrence_rate, 3),
            "DET": round(result.determinism, 3),
            "L": round(result.average_diagonal_line, 3),
            "Lmax": round(result.longest_diagonal_line, 3),
            "DIV": round(result.divergence, 3),
            "ENTR": round(result.entropy_diagonal_lines, 3),
            "LAM": round(result.laminarity, 3),
            "TT": round(result.trapping_time, 3)
        }

        self.RR.append(result.recurrence_rate)
        self.DET.append(result.determinism)
        self.L.append(result.average_diagonal_line)
        self.Lmax.append(result.longest_diagonal_line)
        self.DIV.append(result.divergence)
        self.ENTR.append(result.entropy_diagonal_lines)
        self.LAM.append(result.laminarity)
        self.TT.append(result.trapping_time)

    def update_rqa_plot(self):
        # set up figure
        self.fig_rqa.clear()
        ax_rp_func = self.fig_rqa.add_subplot(111)

        # plot recurrence matrix
        ax_rp_func.plot(self.RR, self.DET)
        # self.fig_rp.colorbar(im, ax=ax_rp_func)
        ax_rp_func.set_title("RQA Measures")
        ax_rp_func.set_xlabel("Recurrence Rate")
        ax_rp_func.set_ylabel("Determinism")

        # display rp
        self.canvas_rp.draw()




    # def start_func(self):
    #     if not self.is_running:
    #         self.is_running = True
    #         self.toggle_buttons("disabled")  # deactivates all user inputs
    #
    #         # set initial conditions if time series is empty, otherwise do nothing and plotting will continue from pause
    #         if len(self.xyzs) == 0:
    #             self.xyzs = np.array(
    #                 [[self.init_cond_x_var.get(), self.init_cond_y_var.get(), self.init_cond_z_var.get()]])
    #
    #         # start plotting
    #         self.update_plot()
    #
    # def stop_func(self):
    #     if self.is_running:
    #         self.is_running = False
    #         self.toggle_buttons("normal")  # reactivates all user inputs
    #
    # def reset_func(self):
    #     self.is_running = False
    #     self.xyzs = np.array([])
    #     self.fig_ps_func.clear()
    #     self.fig_comp_func.clear()
    #     self.fig_rp_func.clear()
    #     self.canvas_ps_func.draw()
    #     self.canvas_comp_func.draw()
    #     self.canvas_rp_func.draw()
    #     self.rqa_measures = {
    #         "RR": 'TBD',
    #         "DET": 'TBD',
    #         "L": 'TBD',
    #         "Lmax": 'TBD',
    #         "DIV": 'TBD',
    #         "ENTR": 'TBD',
    #         "LAM": 'TBD',
    #         "TT": 'TBD'
    #     }
    #     self.display_rqa_measures()
    #     if sum([self.check_var_x.get(),self.check_var_y.get(),self.check_var_z.get()]) == 2:
    #         self.check_button_CRP.configure(state='normal')
    #     else:
    #         self.check_button_CRP.configure(state='disabled')
    #         self.check_var_CRP.set(False)
