from tkinter import ttk
import tkinter as tk

class homeTab:
    def __init__(self, notebook):
        self.notebook = notebook
        self.home_tab = ttk.Frame(notebook)
        notebook.add(self.home_tab, text='Main Menu')
        self.create_home_tab()
    def create_home_tab(self):
        self.title_label = ttk.Label(self.home_tab, text="Welcome to Live Data and Recurrence Plot GUI", font=("Helvetica", 16))
        self.title_label.pack(pady=10)

        self.info_text = tk.Text(self.home_tab, wrap='word', height=10, width=50)
        self.info_text.insert(tk.END,
                              "This GUI allows you to visualize different chaotic systems and analyze their recurrence plots. "
                              "You can start/stop/reset the live plotting of selected functions and see the recurrence quantification analysis (RQA) measures.")
        self.info_text.config(state=tk.DISABLED)
        self.info_text.pack(pady=10)

        # self.btn_functions_tab = ttk.Button(self.home_tab, text="Go to Plotting Functions", command=lambda: self.notebook.select(self.function_tab))
        # self.btn_functions_tab.pack(pady=5)
        #
        # self.btn_data_tab = ttk.Button(self.home_tab, text="Go to Plotting Data", command=lambda: self.notebook.select(self.data_tab))
        # self.btn_data_tab.pack(pady=5)