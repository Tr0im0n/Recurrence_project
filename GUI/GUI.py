import tkinter as tk
from tkinter import ttk
from home_tab import homeTab
from function_tab import funcTab
from data_tab import dataTab

class LivePlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Data and Recurrence Plot")
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=1, fill='both')
        self.home_tab = homeTab(self.notebook)
        self.func_tab = funcTab(self.root, self.notebook)
        self.data_tab = dataTab(self.root, self.notebook)

if __name__ == "__main__":
    root = tk.Tk()
    app = LivePlotApp(root)
    root.mainloop()