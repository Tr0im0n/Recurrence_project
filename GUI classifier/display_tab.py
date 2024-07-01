import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.computation import RQAComputation
from pyrqa.metric import EuclideanMetric
from pyrqa.neighbourhood import FixedRadius
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.preprocessing import StandardScaler
from data_feed import make_window
from time import time
from scipy.stats import entropy
import joblib
import threading


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
        self.is_running = False
        self.path = '/Users/martina/Documents/GitHub/Recurrence_project/datasets/classefiergui.csv'
        self.RR = []
        self.DET = []
        self.L = []
        self.Lmax=[]
        self.DIV = []
        self.ENTR = []
        self.LAM = []
        self.TT = []
        self.fault_detected = 0
        self.fault_memory = 0
        self.fault_count = 0

        # load classifier
        self.classifier = joblib.load('/Users/martina/Documents/GitHub/Recurrence_project/classifier.joblib')
        self.scaler = joblib.load('/Users/martina/Documents/GitHub/Recurrence_project/scaler.joblib')

        # create display tab
        self.disp_tab = tk.Frame(notebook, background='white')
        notebook.add(self.disp_tab, text='Live Window')
        self.create_disp_tab()

        # create menu bar
        self.create_menu_bar()


    def create_menu_bar(self):
        menu_bar = tk.Menu(self.root)

        # Create the "File" menu
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Start", command=self.start_func)
        file_menu.add_command(label="Stop", command=self.stop_func)
        # file_menu.add_command(label='Reset', command=self.reset_func)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Add the "File" menu to the menu bar
        menu_bar.add_cascade(label="File", menu=file_menu)

        # Configure the menu bar
        self.root.config(menu=menu_bar)

    def create_disp_tab(self):
        self.left_frame = tk.Frame(self.disp_tab, background='white')
        self.left_frame.place(relx=0, rely=0, relwidth=0.5, relheight=1)

        self.right_frame = tk.Frame(self.disp_tab, background='white')
        self.right_frame.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)

        self.left_window_layout()
        self.right_window_layout()

        self.update_data_plot()

    def left_window_layout(self):
        # set up live plot figure
        self.live_plot_frame = tk.Frame(self.left_frame, background='white')
        self.live_plot_frame.place(relx=0, rely=0, relwidth=1, relheight=0.5)

        self.fig_live_data = plt.Figure()
        self.canvas_data = FigureCanvasTkAgg(self.fig_live_data, master=self.live_plot_frame)
        self.canvas_data.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

        # set up maintenance warning display
        self.feedback_frame = tk.Frame(self.left_frame, background='white')
        self.feedback_frame.place(relx=0.1, rely=0.55, relwidth=0.8, relheight=0.4)

        self.warning_frame = tk.Frame(self.feedback_frame, background='white', bd=2, relief='solid', highlightbackground="black", highlightcolor="black",
                         highlightthickness=1)
        self.warning_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.fault_log_frame = tk.Frame(self.feedback_frame, background='white')
        self.fault_log_frame.pack(fill='x', padx=10)

        self.fault_count_label = tk.Label(self.feedback_frame, text='Total number of faults detected: ')
        self.fault_count_label.pack(expand=True, fill='x', padx=10)

        self.feedback_frame_layout()

    def update_data_plot(self):
        if not self.is_running:
            return
        # determine elapsed time to extract relevant data window
        current_time = time()
        elapsed_time = round(current_time - self.start_time, 1)

        # get data window
        window_size = 1000
        self.data, self.labels = make_window(self.path, elapsed_time, window_size)

        colors = {1: 'green', 2: 'blue', 3: 'red'}

        # plot phase space of function
        self.fig_live_data.clear()
        ax_ps = self.fig_live_data.add_subplot(111)

        # plot points with color corresponding to error types
        for label in colors:
            label_data = self.data[self.labels == label]
            ax_ps.plot(label_data.index, label_data, c=colors[label], lw=0.5, label=f'Label {label}')

        # ax_ps.plot(t, self.data,lw=0.5)
        ax_ps.set_title("Live data")
        ax_ps.set_xlabel("Time")
        ax_ps.set_ylabel("Data")
        ax_ps.set_ylim(-1, 1)
        self.fig_live_data.tight_layout()
        self.fig_live_data.savefig('live_data.png')

        self.canvas_data.draw()

        if self.is_running:
            self.root.after(200, self.update_data_plot)
            self.update_rp()
            self.update_rqa_plot()
            self.classifier_result()
            self.change_lights()

    def feedback_frame_layout(self):
        # frame title
        warning_title = tk.Label(self.warning_frame, text="Live Machine Status", fg="black", bg="white", font=("Arial", 24))
        warning_title.pack()
        # set up red/green light
        self.light_canvas = tk.Canvas(self.warning_frame, width = 175, height=80, background='white', highlightthickness=0)
        self.light_canvas.pack()

        self.green_light = self.light_canvas.create_oval(25, 15, 75, 65)
        self.light_canvas.itemconfig(self.green_light, fill='gray')
        self.red_light = self.light_canvas.create_oval(100, 15, 150, 65)
        self.light_canvas.itemconfig(self.red_light, fill='gray')

        self.warning_message = tk.StringVar(value='Machine is happy')

        self.warning_label = tk.Label(self.warning_frame, fg="black", bg="white", font=("Arial", 20), textvariable=self.warning_message)
        self.warning_label.pack()

        # Create fault log
        title_label = tk.Label(self.fault_log_frame, text="Fault History", font=("Arial", 14, "bold"))
        title_label.pack(side=tk.TOP, fill='x')

        self.fault_log_box = tk.Text(self.fault_log_frame, wrap=tk.WORD, width=50, height=15)
        self.fault_log_box.pack(fill=tk.BOTH, expand=True)





    def change_lights(self):
        if self.fault_detected == 0:
            self.light_canvas.itemconfig(self.red_light, fill='gray')
            self.light_canvas.itemconfig(self.green_light, fill='green')
            self.warning_message.set('Machine is happy')
        elif self.fault_detected == 1:
            self.light_canvas.itemconfig(self.red_light, fill='red')
            self.light_canvas.itemconfig(self.green_light, fill='gray')
            self.warning_message.set('Inner Race Fault')
        elif self.fault_detected == 2:
            self.light_canvas.itemconfig(self.red_light, fill='red')
            self.light_canvas.itemconfig(self.green_light, fill='gray')
        elif self.fault_detected == 3:
            self.light_canvas.itemconfig(self.red_light, fill='red')
            self.light_canvas.itemconfig(self.green_light, fill='gray')
            self.warning_message.set('Outer Race Fault')


    def right_window_layout(self):
        # set up recurrence plot figure
        self.rp_frame = tk.Frame(self.right_frame, background='white')
        self.rp_frame.place(relx=0, rely=0, relwidth=1, relheight=0.5)

        self.fig_rp = plt.Figure()
        self.canvas_rp = FigureCanvasTkAgg(self.fig_rp, master=self.rp_frame)
        self.canvas_rp.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

        # set up RQA feature plot
        self.rqa_frame = tk.Frame(self.right_frame, background='white')
        self.rqa_frame.place(relx=0, rely=0.5, relwidth=1, relheight=0.5)

        self.fig_rqa = plt.Figure()
        self.canvas_rqa = FigureCanvasTkAgg(self.fig_rqa, master=self.rqa_frame)
        self.canvas_rqa.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

    def update_rp(self):
        # calculates recurrence plot
        self.recurrence_matrix = self.calculate_recurrence_plot()
        # self.calculate_rqa(recurrence_matrix)

        # set up figure
        self.fig_rp.clear()
        ax_rp = self.fig_rp.add_subplot(111)

        # plot recurrence matrix
        im = ax_rp.imshow(self.recurrence_matrix, cmap='binary', origin='lower')
        # self.fig_rp.colorbar(im, ax=ax_rp)
        ax_rp.set_title(f"Recurrence Plot")
        ax_rp.set_xlabel("Vector Index")
        ax_rp.set_ylabel("Vector Index")
        self.fig_rp.tight_layout()
        self.fig_rp.savefig('rp.png')

        # display rp
        self.canvas_rp.draw()

    def calculate_recurrence_plot(self):
        self.m = 5
        self.T = 2
        self.epsilon = 0.3

        # embed time series
        self.num_vectors = len(self.data) - (self.m - 1) * self.T
        self.vectors = np.array([self.data[t:t + self.m * self.T:self.T] for t in range(self.num_vectors)])

        if self.vectors.size > 0:  # check that enough points exist to create recurrence plot
            # create and normalize similarity matrix
            self.D = squareform(pdist(self.vectors, metric='euclidean'))
            D_max = np.max(self.D)
            self.D_norm = self.D / D_max
            recurrence_matrix = self.D_norm < self.epsilon
            return recurrence_matrix


    def calculate_rqa_measures_pyrqa(self):
        time_series = TimeSeries(self.data, embedding_dimension=self.m, time_delay=self.T)
        settings = Settings(
            time_series,
            neighbourhood=FixedRadius(self.epsilon),
            similarity_measure=EuclideanMetric(),
            theiler_corrector=1
        )

        computation = RQAComputation.create(settings)
        result = computation.run()

        self.rqa_measures = np.array([result.recurrence_rate,
                                      result.determinism,
                                      result.average_diagonal_line,
                                      result.trapping_time,
                                      result.longest_diagonal_line,
                                      result.divergence,
                                      result.entropy_diagonal_lines,
                                      result.laminarity]).reshape(1, -1)

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
        ax_rqa_func = self.fig_rqa.add_subplot(111)

        thread = threading.Thread(target=self.calculate_rqa_measures_pyrqa())
        thread.start()

        self.calculate_rqa_measures_pyrqa()

        # plot recurrence matrix
        ax_rqa_func.scatter(self.RR, self.DET)

        # self.fig_rp.colorbar(im, ax=ax_rp_func)
        ax_rqa_func.set_title("RQA Measures")
        ax_rqa_func.set_xlabel("Recurrence Rate")
        ax_rqa_func.set_ylabel("Determinism")

        # display rp
        self.canvas_rqa.draw()

    def classifier_result(self):
        self.rqa_measures_scaled = self.scaler.transform(self.rqa_measures)
        self.fault_detected = self.classifier.predict(self.rqa_measures_scaled.reshape(1, -1))
        self.detection_time = round(time() - self.start_time, 2)

        # Initialize counter for number of consecutive faults detected
        if not hasattr(self, 'consecutive_fault_count'):
            self.consecutive_fault_count = 0

        # Initialize variable to track type of last recorded fault
        if not hasattr(self, 'recorded_fault_record'):
            self.recorded_fault_record = 0

        # Check for faults and update the counter
        if self.fault_detected in [1, 2, 3]:
            if self.fault_detected == self.fault_memory:
                self.consecutive_fault_count += 1
            else:
                self.consecutive_fault_count = 1
        else:
            self.consecutive_fault_count = 0

        print(self.fault_detected)
        print('consecutive faults: ', self.consecutive_fault_count)
        print('fault memory: ', self.fault_memory)

        if self.fault_detected != 0 and self.consecutive_fault_count >= 2 and self.fault_detected != self.recorded_fault_record:
            fault_type = ['Inner Race', 'Ball', 'Outer Race']
            fault_index = int(self.fault_detected) - 1
            self.fault_log_box.insert(tk.END,
                                      f'Fault Detected: {fault_type[fault_index]} Fault at {self.detection_time} s' + '\n')
            self.fault_log_box.yview(tk.END)  # Scroll to the end
            self.fault_count += 1
            self.recorded_fault_record = self.fault_detected
        self.fault_memory = self.fault_detected
        # add detected fault to log


    def start_func(self):
        if not self.is_running:
            self.is_running = True
            self.start_time = time() - 2
            self.update_data_plot()

    def stop_func(self):
        if self.is_running:
            self.is_running = False

    def reset_func(self):
        self.is_running = False
        self.fig_live_data.clear()
        self.fig_rp.clear()
        self.fig_rqa.clear()
        self.canvas_data.draw()
        self.canvas_rp.draw()
        self.canvas_rqa.draw()
        self.__init__()
