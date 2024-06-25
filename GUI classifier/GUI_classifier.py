# GUI for predictive maintenance using SVM classification of time series data with recurrence plots
# Maastricht University, Circular Engineering
# June, 2024

# Imports
import tkinter as tk
from tkinter import ttk
from display_tab import dispTab
from settings_tab import settingsTab

class LivePlotApp:
    def __init__(self, root):
        # window set up
        self.root = root
        self.root.title("Predictive Maintenance Interface")
        self.root.configure(bg='white')
        self.root.geometry("1000x800")

        # tab set up
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill='both')
        self.display_tab = dispTab(self.root, self.notebook)
        self.settings_tab = settingsTab(self.root, self.notebook)



if __name__ == "__main__":
    root = tk.Tk()
    app = LivePlotApp(root)
    root.mainloop()