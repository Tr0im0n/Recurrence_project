import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.spatial.distance import pdist, squareform

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter.filedialog as fd

class dispTab:
    def __init__(self, root, notebook):
        # access root and notebook from main file
        self.root = root
        self.notebook = notebook

        self.disp_tab = ttk.Frame(notebook)
        notebook.add(self.disp_tab, text='Live Window')
        self.create_disp_tab()


    def create_disp_tab(self):
        # set up figures
        self.fig_live_data = plt.Figure()
        self.fig_rp = plt.Figure()
        self.fig_rqas = plt.Figure()

        self.canvas_data = FigureCanvasTkAgg(self.fig_rp, master=self.disp_tab)
        self.canvas_data.get_tk_widget().place(relx=0, rely=0, relwidth=0.5, relheight=0.5)

        self.canvas_rp = FigureCanvasTkAgg(self.fig_rqas, master=self.disp_tab)
        self.canvas_rp.get_tk_widget().place(relx=0, rely=0.5, relwidth=0.5, relheight=0.5)


    def update_plot(self):
        if not self.is_running:
            return

        self.data = []

        # plot phase space of function
        self.fig_live_data.clear()
        ax_ps = self.fig_live_data.add_subplot(111)
        ax_ps.plot(*self.data, lw=0.5)
        ax_ps.set_title(f"Live data")
        ax_ps.set_xlabel("Time")
        ax_ps.set_ylabel("Data")
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
            ax_comp.plot(self.x, lw=0.5)
            ax_comp.set_title(f"{coord_key[true_index]}-coordinate")
            ax_comp.set_xlabel("Time")
            ax_comp.set_ylabel(f"{coord_key[true_index]}-coordinate")

            # calculates recurrence plot
            recurrence_matrix = self.calculate_recurrence_plot(self.x)
            self.calculate_rqa(recurrence_matrix)
            print(self.rqa_measures)
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
                    print('data: ', data)

                    # add time series to plot
                    ax_comp.plot(data, lw=0.5, label=f'{coord_key[i]}-coordinate')
                    ax_comp.legend()

                    if first == True:
                        data1 = data
                        first = False
                    else:
                        data2 = data
                        print(data1)
                        print(data2)
                        rp = self.calculate_cross_recurrence_plot(data1, data)

            # self.calculate_rqa(rp)
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