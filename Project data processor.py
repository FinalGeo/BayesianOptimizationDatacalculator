import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk, scrolledtext
from tkinter.simpledialog import Dialog
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation, FFMpegWriter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, RationalQuadratic, ExpSineSquared, DotProduct
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import seaborn as sns
import os
import time
import re
import traceback
import io
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Use TkAgg backend for interactive plots
matplotlib.use('TkAgg')

class CustomSaveDialog(Dialog):
    """Custom dialog for plot customization before saving"""
    def __init__(self, parent, title="Customize Plot", current_title="", current_xlabel="", current_ylabel=""):
        self.current_title = current_title
        self.current_xlabel = current_xlabel
        self.current_ylabel = current_ylabel
        super().__init__(parent, title)
    
    def body(self, master):
        tk.Label(master, text="Plot Title:").grid(row=0, sticky=tk.W, pady=5)
        self.title_entry = tk.Entry(master, width=40)
        self.title_entry.insert(0, self.current_title)
        self.title_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        tk.Label(master, text="X-axis Label:").grid(row=1, sticky=tk.W, pady=5)
        self.xlabel_entry = tk.Entry(master, width=40)
        self.xlabel_entry.insert(0, self.current_xlabel)
        self.xlabel_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        tk.Label(master, text="Y-axis Label:").grid(row=2, sticky=tk.W, pady=5)
        self.ylabel_entry = tk.Entry(master, width=40)
        self.ylabel_entry.insert(0, self.current_ylabel)
        self.ylabel_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        return self.title_entry  # initial focus
    
    def apply(self):
        self.result = {
            'title': self.title_entry.get(),
            'xlabel': self.xlabel_entry.get(),
            'ylabel': self.ylabel_entry.get()
        }

class SelectionDialog(tk.Toplevel):
    """Custom dialog for data selection using Treeview"""
    def __init__(self, parent, data, title="Select Data"):
        super().__init__(parent)
        self.title(title)
        self.data = data
        self.result = None
        self.selected_indices = []
        
        # Set window size
        self.geometry("1000x600")
        
        # Create a frame for the Treeview and scrollbars
        frame = ttk.Frame(self)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create Treeview
        self.tree = ttk.Treeview(frame, show="headings")
        self.tree["columns"] = list(data.columns)
        
        # Setup columns
        for col in data.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor="w")
        
        # Add data
        self.tree.tag_configure('selected', background='lightblue')
        for i, row in data.iterrows():
            values = list(row)
            self.tree.insert("", "end", iid=str(i), values=values)
        
        # Scrollbars
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Layout
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        # Configure grid weights
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        
        # Button frame
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="Apply", command=self.on_apply).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side="left", padx=5)
        
        # Bind selection event
        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        
        # Make modal
        self.transient(parent)
        self.grab_set()
        parent.wait_window(self)
    
    def on_select(self, event):
        """Handle row selection"""
        selected_items = self.tree.selection()
        self.selected_indices = [int(item) for item in selected_items]
        
        # Update background color for selected items
        for item in self.tree.get_children():
            if item in selected_items:
                self.tree.item(item, tags=('selected',))
            else:
                self.tree.item(item, tags=())
    
    def on_apply(self):
        """Return selected rows as DataFrame"""
        if self.selected_indices:
            self.result = self.data.iloc[self.selected_indices].copy()
        self.destroy()

class GPR_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GPR, BO Visualization & Data Analysis")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create GPR tab
        self.gpr_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.gpr_tab, text="Gaussian Process Regression")
        
        # Create BO Visualization tab
        self.bo_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.bo_tab, text="Bayesian Optimization Visualization")
        
        # Create Data Selection tab
        self.data_selection_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.data_selection_tab, text="Data Selection")
        
        # Create Violin Plots tab
        self.violin_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.violin_tab, text="Violin Plots")
        
        # Create Statistics tab
        self.stats_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_tab, text="Statistics")
        
        # Initialize all tabs
        self.create_gpr_tab(self.gpr_tab)
        self.create_bo_tab(self.bo_tab)
        self.create_data_selection_tab(self.data_selection_tab)
        
        # Initialize datasets for violin plots and statistics
        self.stats_data = None
        self.stats_datasets = {
            "Dataset 1": None,
            "Dataset 2": None,
            "Dataset 3": None,
            "Dataset 4": None,
            "Total": None
        }
        self.stats_numeric_vars = []
        self.current_violin_plot_fig = None
        self.current_violin_plot_canvas = None
        self.bo_animation = None
        self.animation_running = False
        
        # Initialize violin and stats tabs (will be populated later)
        self.violin_label = ttk.Label(self.violin_tab, text="Please define datasets in the Data Selection tab first.")
        self.violin_label.pack(pady=50)
        
        self.stats_label = ttk.Label(self.stats_tab, text="Please define datasets in the Data Selection tab first.")
        self.stats_label.pack(pady=50)
    
    def create_gpr_tab(self, parent):
        # Original GPR implementation
        self.data = None
        self.X = None
        self.y = None
        self.random_state = 42
        self.X_scaler = None
        self.y_scaler = None
        self.original_xlim = None
        self.original_ylim = None
        self.current_title = "Gaussian Process Regression"
        self.current_xlabel = "Input Feature"
        self.current_ylabel = "Target Value"
        
        # Create GUI elements for GPR tab
        header = tk.Frame(parent, bg='#2c3e50', height=60)
        header.pack(fill=tk.X, padx=10, pady=10)
        tk.Label(header, text="Gaussian Process Regression", font=('Arial', 16, 'bold'), 
                bg='#2c3e50', fg='white').pack(pady=15)
        
        # Control panel
        control_frame = tk.Frame(parent, bg='#ecf0f1', padx=10, pady=10)
        control_frame.pack(fill=tk.X, padx=10, pady=(0,10))
        
        # Dataset selection
        tk.Label(control_frame, text="Dataset:", bg='#ecf0f1', font=('Arial', 10)).pack(side=tk.LEFT, padx=(0,5))
        self.dataset_var = tk.StringVar()
        self.dataset_dropdown = tk.OptionMenu(control_frame, self.dataset_var, "")
        self.dataset_dropdown.config(width=15)
        self.dataset_dropdown.pack(side=tk.LEFT, padx=5)
        self.dataset_dropdown.config(state=tk.DISABLED)
        
        # Process button
        self.process_btn = tk.Button(control_frame, text="Run GPR", command=self.run_gpr,
                                   bg='#2ecc71', fg='white', font=('Arial', 10, 'bold'), width=10, state=tk.DISABLED)
        self.process_btn.pack(side=tk.LEFT, padx=5)
        
        # Kernel selection
        tk.Label(control_frame, text="Kernel:", bg='#ecf0f1', font=('Arial', 10)).pack(side=tk.LEFT, padx=(10,5))
        self.kernel_var = tk.StringVar()
        self.kernel_var.set("Matern 5/2")  # Default to Matern 5/2
        kernel_menu = tk.OptionMenu(control_frame, self.kernel_var, 
                                   "RBF", "Matern 3/2", "Matern 5/2", 
                                   "RationalQuadratic", "RBF+Matern", 
                                   "Periodic", "Linear")
        kernel_menu.config(width=12)
        kernel_menu.pack(side=tk.LEFT, padx=5)
        
        # Optimization control
        tk.Label(control_frame, text="Restarts:", bg='#ecf0f1', font=('Arial', 10)).pack(side=tk.LEFT, padx=(10,5))
        self.restart_var = tk.StringVar(value="30")
        restart_spin = tk.Spinbox(control_frame, from_=10, to=100, increment=5, 
                                 textvariable=self.restart_var, width=5)
        restart_spin.pack(side=tk.LEFT, padx=5)
        
        # Save plot button
        self.save_btn = tk.Button(control_frame, text="Save Plot", command=self.custom_save,
                                 bg='#9b59b6', fg='white', font=('Arial', 10, 'bold'), width=10, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # Home button
        self.home_btn = tk.Button(control_frame, text="Reset View", command=self.reset_view,
                                 bg='#e67e22', fg='white', font=('Arial', 10, 'bold'), width=10, state=tk.DISABLED)
        self.home_btn.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status = tk.StringVar()
        self.status.set("Please import data to begin")
        tk.Label(control_frame, textvariable=self.status, bg='#ecf0f1', font=('Arial', 10)).pack(side=tk.LEFT, padx=10)
        
        # Plot area
        plot_frame = tk.Frame(parent, bg='white', bd=2, relief=tk.SUNKEN)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))
        
        # Create figure
        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title(self.current_title, fontsize=14)
        self.ax.set_xlabel(self.current_xlabel, fontsize=12)
        self.ax.set_ylabel(self.current_ylabel, fontsize=12)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create navigation toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Info panel
        info_frame = tk.Frame(parent, bg='#ecf0f1', padx=10, pady=10)
        info_frame.pack(fill=tk.X, padx=10, pady=(0,10))
        
        # Kernel info
        kernel_frame = tk.Frame(info_frame, bg='#ecf0f1')
        kernel_frame.pack(fill=tk.X, pady=(0,5))
        tk.Label(kernel_frame, text="Optimized Kernel:", bg='#ecf0f1', font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        self.kernel_info = tk.StringVar()
        self.kernel_info.set("None")
        tk.Label(kernel_frame, textvariable=self.kernel_info, bg='#ecf0f1', font=('Courier', 9)).pack(side=tk.LEFT, padx=5)
        
        # Likelihood info
        lml_frame = tk.Frame(info_frame, bg='#ecf0f1')
        lml_frame.pack(fill=tk.X, pady=(0,5))
        tk.Label(lml_frame, text="Log-Marginal-Likelihood:", bg='#ecf0f1', font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        self.lml_info = tk.StringVar()
        self.lml_info.set("None")
        tk.Label(lml_frame, textvariable=self.lml_info, bg='#ecf0f1', font=('Courier', 9)).pack(side=tk.LEFT, padx=5)
        
        # Column selection
        col_frame = tk.Frame(parent, bg='#ecf0f1', padx=10, pady=5)
        col_frame.pack(fill=tk.X, padx=10, pady=(0,10))
        
        tk.Label(col_frame, text="X Column:", bg='#ecf0f1', font=('Arial', 10)).pack(side=tk.LEFT, padx=(0,5))
        self.x_var = tk.StringVar()
        self.x_dropdown = tk.OptionMenu(col_frame, self.x_var, "")
        self.x_dropdown.config(width=15)
        self.x_dropdown.pack(side=tk.LEFT, padx=5)
        
        tk.Label(col_frame, text="Y Column:", bg='#ecf0f1', font=('Arial', 10)).pack(side=tk.LEFT, padx=(10,5))
        self.y_var = tk.StringVar()
        self.y_dropdown = tk.OptionMenu(col_frame, self.y_var, "")
        self.y_dropdown.config(width=15)
        self.y_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Sample size control
        tk.Label(col_frame, text="Max Points:", bg='#ecf0f1', font=('Arial', 10)).pack(side=tk.LEFT, padx=(20,5))
        self.max_points_var = tk.StringVar(value="1000")
        max_points_spin = tk.Spinbox(col_frame, from_=100, to=10000, increment=100, 
                                    textvariable=self.max_points_var, width=6)
        max_points_spin.pack(side=tk.LEFT, padx=5)
        
        # Label customization button
        tk.Button(col_frame, text="Customize Labels", command=self.customize_labels,
                 bg='#f1c40f', fg='black', font=('Arial', 9, 'bold'), width=15).pack(side=tk.RIGHT, padx=10)
        
        self.x_dropdown.config(state=tk.DISABLED)
        self.y_dropdown.config(state=tk.DISABLED)
        
        # Trace variable changes
        self.x_var.trace_add("write", self.update_plot_from_selection)
        self.y_var.trace_add("write", self.update_plot_from_selection)
        self.dataset_var.trace_add("write", self.update_dataset_selection)
    
    def update_dataset_selection(self, *args):
        """Update GPR when dataset is changed"""
        dataset_name = self.dataset_var.get()
        if dataset_name and dataset_name in self.stats_datasets and self.stats_datasets[dataset_name] is not None:
            self.data = self.stats_datasets[dataset_name]
            self.update_column_dropdowns()
            
            # Set default selections if possible
            if self.data.columns.size >= 2:
                self.x_var.set(self.data.columns[0])
                self.y_var.set(self.data.columns[1])
            
            point_count = len(self.data)
            self.status.set(f"Selected: {dataset_name} | Points: {point_count}")
            self.process_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)
            self.home_btn.config(state=tk.NORMAL)
            self.plot_data()
        else:
            self.status.set("No valid dataset selected")
    
    def update_column_dropdowns(self):
        # Clear current dropdown menus
        self.x_dropdown['menu'].delete(0, 'end')
        self.y_dropdown['menu'].delete(0, 'end')
        
        # Add new columns
        if self.data is not None:
            self.x_dropdown.config(state=tk.NORMAL)
            self.y_dropdown.config(state=tk.NORMAL)
            for col in self.data.columns:
                self.x_dropdown['menu'].add_command(label=col, command=tk._setit(self.x_var, col))
                self.y_dropdown['menu'].add_command(label=col, command=tk._setit(self.y_var, col))
        else:
            self.x_dropdown.config(state=tk.DISABLED)
            self.y_dropdown.config(state=tk.DISABLED)
    
    def update_plot_from_selection(self, *args):
        if self.data is not None:
            self.plot_data()
    
    def plot_data(self):
        self.ax.clear()
        if self.data is not None and self.x_var.get() and self.y_var.get():
            try:
                x_col = self.x_var.get()
                y_col = self.y_var.get()
                
                # Create a copy to avoid modifying original data
                plot_data = self.data[[x_col, y_col]].copy()
                
                # Convert to numeric, coercing errors
                plot_data[x_col] = pd.to_numeric(plot_data[x_col], errors='coerce')
                plot_data[y_col] = pd.to_numeric(plot_data[y_col], errors='coerce')
                
                # Drop rows with NaN
                plot_data = plot_data.dropna()
                
                if len(plot_data) == 0:
                    raise ValueError("No numeric data available after conversion")
                
                # Apply max points limit
                max_points = int(self.max_points_var.get())
                if len(plot_data) > max_points:
                    plot_data = plot_data.sample(n=max_points, random_state=self.random_state)
                
                X = plot_data[x_col].values.reshape(-1, 1)
                y = plot_data[y_col].values
                
                self.ax.scatter(X, y, c='k', s=10, zorder=10, label='Observations', alpha=0.7)
                self.ax.set_title(self.current_title, fontsize=14)
                self.ax.set_xlabel(self.current_xlabel, fontsize=12)
                self.ax.set_ylabel(self.current_ylabel, fontsize=12)
                self.ax.grid(True, alpha=0.3)
                self.ax.legend(loc='best')
                
                # Store original view limits
                self.original_xlim = self.ax.get_xlim()
                self.original_ylim = self.ax.get_ylim()
                
            except Exception as e:
                self.ax.text(0.5, 0.5, f'Error displaying data\n{str(e)}',
                             horizontalalignment='center', verticalalignment='center',
                             transform=self.ax.transAxes, fontsize=12, color='red')
        else:
            self.ax.text(0.5, 0.5, 'No data available\nSelect a dataset and columns',
                         horizontalalignment='center', verticalalignment='center',
                         transform=self.ax.transAxes, fontsize=12)
        
        self.canvas.draw()
    
    def create_kernel(self):
        """Create kernel based on user selection with appropriate bounds"""
        kernel_type = self.kernel_var.get()
        min_ls = 0.1  # Minimum length scale
        max_ls = 10.0  # Maximum length scale
        
        if kernel_type == "RBF":
            return C(1.0, (1e-3, 1e3)) * RBF(1.0, (min_ls, max_ls))
            
        elif kernel_type == "Matern 3/2":
            return C(1.0, (1e-3, 1e3)) * Matern(1.0, (min_ls, max_ls), nu=1.5)
            
        elif kernel_type == "Matern 5/2":
            return C(1.0, (1e-3, 1e3)) * Matern(1.0, (min_ls, max_ls), nu=2.5)
            
        elif kernel_type == "RationalQuadratic":
            return C(1.0, (1e-3, 1e3)) * RationalQuadratic(1.0, (min_ls, max_ls))
            
        elif kernel_type == "RBF+Matern":
            return (C(1.0, (1e-3, 1e3)) * RBF(1.0, (min_ls, max_ls)) + 
                    C(1.0, (1e-3, 1e3)) * Matern(1.0, (min_ls, max_ls), nu=2.5))
        
        elif kernel_type == "Periodic":
            return C(1.0, (1e-3, 1e3)) * ExpSineSquared(1.0, 5.0, periodicity_bounds=(1, 20))
            
        elif kernel_type == "Linear":
            return C(1.0, (1e-3, 1e3)) * DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0))
        
        else:
            return C(1.0, (1e-3, 1e3)) * RBF(1.0, (min_ls, max_ls))
    
    def run_gpr(self):
        if self.data is None or not self.x_var.get() or not self.y_var.get():
            return
            
        try:
            # Get selected columns
            x_col = self.x_var.get()
            y_col = self.y_var.get()
            
            # Create a copy to avoid modifying original data
            data_subset = self.data[[x_col, y_col]].copy()
            
            # Convert to numeric, coercing errors
            data_subset[x_col] = pd.to_numeric(data_subset[x_col], errors='coerce')
            data_subset[y_col] = pd.to_numeric(data_subset[y_col], errors='coerce')
            
            # Drop rows with NaN
            data_subset = data_subset.dropna()
            
            if len(data_subset) == 0:
                raise ValueError("No numeric data available for the selected columns")
            
            # Apply max points limit
            max_points = int(self.max_points_var.get())
            if len(data_subset) > max_points:
                data_subset = data_subset.sample(n=max_points, random_state=self.random_state)
            
            X = data_subset[x_col].values.reshape(-1, 1)
            y = data_subset[y_col].values
            
            # Add small noise to constant values to avoid kernel warnings
            if np.all(y == y[0]):
                y += np.random.normal(0, 1e-5, len(y))
            
            # Scale features and target
            self.X_scaler = StandardScaler()
            self.y_scaler = StandardScaler()
            
            X_scaled = self.X_scaler.fit_transform(X)
            y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
            
            # Create kernel
            kernel = self.create_kernel()
            
            # Get number of optimization restarts
            n_restarts = int(self.restart_var.get())
            
            # Create and fit GPR model
            gp = GaussianProcessRegressor(
                kernel=kernel, 
                n_restarts_optimizer=n_restarts,
                random_state=self.random_state,
                alpha=1e-5,  # Small noise for numerical stability
                normalize_y=True  # Normalize target internally
            )
            gp.fit(X_scaled, y_scaled)
            
            # Generate predictions
            x_min, x_max = X.min(), X.max()
            x_range = x_max - x_min
            X_pred = np.linspace(x_min - 0.1*x_range, x_max + 0.1*x_range, 500).reshape(-1, 1)
            
            # Scale prediction inputs
            X_pred_scaled = self.X_scaler.transform(X_pred)
            
            # Predict on scaled data
            y_pred_scaled, sigma_scaled = gp.predict(X_pred_scaled, return_std=True)
            
            # Unscale predictions
            y_pred = self.y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            sigma = sigma_scaled * self.y_scaler.scale_
            
            # Update plot
            self.ax.clear()
            
            # Plot a subset of points for large datasets
            plot_points = min(1000, len(X))
            if len(X) > 1000:
                indices = np.random.choice(len(X), size=plot_points, replace=False)
                self.ax.scatter(X[indices], y[indices], c='k', s=10, 
                               zorder=10, label='Observations', alpha=0.5)
            else:
                self.ax.scatter(X, y, c='k', s=10, zorder=10, 
                               label='Observations', alpha=0.5)
                
            self.ax.plot(X_pred, y_pred, 'b-', linewidth=2, label='Prediction')
            self.ax.fill_between(X_pred.ravel(), 
                                y_pred - 1.96*sigma, 
                                y_pred + 1.96*sigma,
                                alpha=0.2, color='blue', label='95% CI')
            self.ax.set_title(self.current_title, fontsize=14)
            self.ax.set_xlabel(self.current_xlabel, fontsize=12)
            self.ax.set_ylabel(self.current_ylabel, fontsize=12)
            self.ax.grid(True, alpha=0.3)
            self.ax.legend(loc='best')
            
            # Store original view limits
            self.original_xlim = self.ax.get_xlim()
            self.original_ylim = self.ax.get_ylim()
            
            self.canvas.draw()
            
            # Update kernel info
            self.kernel_info.set(f"{gp.kernel_}")
            self.lml_info.set(f"{gp.log_marginal_likelihood_value_:.3f}")
            
            self.status.set(f"GPR completed on {len(X)} points using {self.kernel_var.get()} kernel")
            
        except Exception as e:
            messagebox.showerror("Error", f"GPR failed:\n{str(e)}")
    
    def reset_view(self):
        """Reset the plot to its original view"""
        if self.original_xlim and self.original_ylim:
            self.ax.set_xlim(self.original_x极)
            self.ax.set_ylim(self.original_ylim)
            self.canvas.draw()
    
    def customize_labels(self):
        """Open dialog to customize plot labels"""
        dialog = CustomSaveDialog(
            self.root, 
            title="Customize Plot Labels",
            current_title=self.current_title,
            current_xlabel=self.current_xlabel,
            current_ylabel=self.current_ylabel
        )
        
        if hasattr(dialog, 'result'):
            self.current_title = dialog.result['title']
            self.current_xlabel = dialog.result['xlabel']
            self.current_ylabel = dialog.result['ylabel']
            
            # Apply changes to current plot
            self.ax.set_title(self.current_title)
            self.ax.set_xlabel(self.current_xlabel)
            self.ax.set_ylabel(self.current_ylabel)
            self.canvas.draw()
    
    def custom_save(self):
        """Save plot with custom title and labels"""
        if self.data is None:
            return
            
        # Open customization dialog
        dialog = CustomSaveDialog(
            self.root, 
            title="Customize Plot Before Saving",
            current_title=self.current_title,
            current_xlabel=self.current_xlabel,
            current_ylabel=self.current_ylabel
        )
        
        if not hasattr(dialog, 'result'):
            return  # User canceled
        
        # Get file path
        file_path = filedialog.asksaveasfilename(
            title="Save Plot As",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("SVG files", "*.svg"), ("All files", "*.*")],
            defaultextension=".png"
        )
        
        if not file_path:
            return
            
        try:
            # Remember current labels
            original_title = self.ax.get_title()
            original_xlabel = self.ax.get_xlabel()
            original_ylabel = self.ax.get_ylabel()
            
            # Apply custom labels
            self.ax.set_title(dialog.result['title'])
            self.ax.set_xlabel(dialog.result['xlabel'])
            self.ax.set_ylabel(dialog.result['ylabel'])
            
            # Draw changes temporarily
            self.canvas.draw()
            
            # Save with custom labels
            self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            self.status.set(f"Plot saved to: {file_path}")
            
            # Restore original labels
            self.ax.set_title(original_title)
            self.ax.set_xlabel(original_xlabel)
            self.ax.set_ylabel(original_ylabel)
            self.canvas.draw()
            
            # Update persistent labels
            self.current_title = dialog.result['title']
            self.current_xlabel = dialog.result['xlabel']
            self.current_ylabel = dialog.result['ylabel']
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save plot:\n{str(e)}")
    
    def create_bo_tab(self, parent):
        # BO Visualization implementation
        self.bo_data = None
        self.input_cols = []
        self.output_cols = []
        self.pca = None
        self.X_pca = None
        self.score = None
        self.best_idx = None
        self.bo_animation = None
        self.animation_running = False
        self.star_offset_x = 0.05
        self.star_offset_y = 0.05
        self.cmap = None  # Store colormap for consistency
        
        # Control panel
        control_frame = tk.Frame(parent, bg='#ecf0f1', padx=10, pady=10)
        control_frame.pack(fill=tk.X, padx=10, pady=(0,10))
        
        # Dataset selection
        tk.Label(control_frame, text="Dataset:", bg='#ecf0f1', font=('Arial', 10)).pack(side=tk.LEFT, padx=(0,5))
        self.bo_dataset_var = tk.StringVar()
        self.bo_dataset_dropdown = tk.OptionMenu(control_frame, self.bo_dataset_var, "")
        self.bo_dataset_dropdown.config(width=15)
        self.bo_dataset_dropdown.pack(side=tk.LEFT, padx=5)
        self.bo_dataset_dropdown.config(state=tk.DISABLED)
        
        # Input columns selection
        tk.Label(control_frame, text="Input Columns:", bg='#ecf0f1', font=('Arial', 10)).pack(side=tk.LEFT, padx=(10,5))
        self.input_var = tk.StringVar()
        self.input_dropdown = tk.OptionMenu(control_frame, self.input_var, "")
        self.input_dropdown.config(width=20)
        self.input_dropdown.pack(side=tk.LEFT, padx=5)
        self.input_dropdown.config(state=tk.DISABLED)
        
        # Output columns selection
        tk.Label(control_frame, text="Output Columns:", bg='#ecf0f1', font=('Arial', 10)).pack(side=tk.LEFT, padx=(10,5))
        self.output_var = tk.StringVar()
        self.output_dropdown = tk.OptionMenu(control_frame, self.output_var, "")
        self.output_dropdown.config(width=20)
        self.output_dropdown.pack(side=tk.LEFT, padx=5)
        self.output_dropdown.config(state=tk.DISABLED)
        
        # Weights input
        weights_frame = tk.Frame(control_frame, bg='#ecf0f1')
        weights_frame.pack(side=tk.LEFT, padx=(20,5))
        
        tk.Label(weights_frame, text="Weights (w1,w2,w3):", bg='#ecf0f1', font=('Arial', 10)).pack(side=tk.LEFT)
        self.weights_var = tk.StringVar(value="1.0,1.0,1.0")
        weights_entry = tk.Entry(weights_frame, textvariable=self.weights_var, width=15)
        weights_entry.pack(side=tk.LEFT, padx=5)
        
        # Process button
        self.bo_process_btn = tk.Button(control_frame, text="Visualize BO", command=self.visualize_bo,
                                   bg='#2ecc71', fg='white', font=('Arial', 10, 'bold'), width=15, state=tk.DISABLED)
        self.bo_process_btn.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.bo_status = tk.StringVar()
        self.bo_status.set("Please select a dataset")
        tk.Label(control_frame, textvariable=self.bo_status, bg='#ecf0f1', font=('Arial', 10)).pack(side=tk.RIGHT, padx=10)
        
        # Animation controls
        anim_frame = tk.Frame(parent, bg='#ecf0f1', padx=10, pady=5)
        anim_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.anim_btn = tk.Button(anim_frame, text="Start Animation", command=self.toggle_animation,
                                 bg='#3498db', fg='white', font=('Arial', 10, 'bold'), width=15, state=tk.DISABLED)
        self.anim_btn.pack(side=tk.LEFT, padx=5)
        
        tk.Label(anim_frame, text="Speed:", bg='#ecf0f1', font=('Arial', 10)).pack(side=tk.LEFT, padx=(20,5))
        self.speed_var = tk.DoubleVar(value=0.5)
        speed_scale = tk.Scale(anim_frame, from_=0.1, to=2.0, resolution=0.1, orient=tk.HORIZONTAL,
                              variable=self.speed_var, length=150, showvalue=True)
        speed_scale.pack(side=tk.LEFT, padx=5)
        
        # Save animation button
        self.save_anim_btn = tk.Button(anim_frame, text="Save Animation", command=self.save_animation,
                                     bg='#9b59b6', fg='white', font=('Arial', 10, 'bold'), width=15, state=tk.DISABLED)
        self.save_anim_btn.pack(side=tk.LEFT, padx=5)
        
        # Reset view button
        self.reset_btn = tk.Button(anim_frame, text="Reset View", command=self.reset_bo_view,
                                 bg='#e67e22', fg='white', font=('Arial', 10, 'bold'), width=10, state=tk.DISABLED)
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        
        # Star offset controls
        offset_frame = tk.Frame(anim_frame, bg='#ecf0f1')
        offset_frame.pack(side=tk.LEFT, padx=(20,0))
        
        tk.Label(offset_frame, text="Star Offset:", bg='#ecf0f1', font=('Arial', 10)).pack(side=tk.LEFT)
        
        tk.Label(offset_frame, text="X:", bg='#ecf0f1', font=('Arial', 9)).pack(side=tk.LEFT, padx=(5,0))
        self.star_offset_x_var = tk.DoubleVar(value=self.star_offset_x)
        offset_x_spin = tk.Spinbox(offset_frame, from_=0.01, to=0.2, increment=0.01, 
                                  textvariable=self.star_offset_x_var, width=4)
        offset_x_spin.pack(side=tk.LEFT)
        
        tk.Label(offset_frame, text="Y:", bg='#ecf0f1', font=('Arial', 9)).pack(side=tk.LEFT, padx=(5,0))
        self.star_offset_y_var = tk.DoubleVar(value=self.star_offset_y)
        offset_y_spin = tk.Spinbox(offset_frame, from_=0.01, to=0.2, increment=0.01, 
                                  textvariable=self.star_offset_y_var, width=4)
        offset_y_spin.pack(side=tk.LEFT)
        
        # Apply offset button
        tk.Button(offset_frame, text="Apply", command=self.apply_star_offset,
                 bg='#f1c40f', fg='black', font=('Arial', 9, 'bold'), width=5).pack(side=tk.LEFT, padx=5)
        
        # Plot area
        plot_frame = tk.Frame(parent, bg='white', bd=2, relief=tk.SUNKEN)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))
        
        # Create figure with consistent size
        self.bo_fig = Figure(figsize=(10, 8), dpi=100)  # Larger fixed size
        self.bo_ax = self.bo_fig.add_subplot(111)
        self.bo_ax.grid(True, alpha=0.3)
        self.bo_ax.set_title("Bayesian Optimization Visualization", fontsize=14)
        self.bo_ax.set_xlabel("PC1 (Summary of Inputs)", fontsize=12)
        self.bo_ax.set_ylabel("PC2 (Another Summary)", fontsize=12)
        
        # Add padding to prevent layout shifts
        self.bo_fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        
        # Create canvas
        self.bo_canvas = FigureCanvasTkAgg(self.bo_fig, master=plot_frame)
        self.bo_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create navigation toolbar
        self.bo_toolbar = NavigationToolbar2Tk(self.bo_canvas, plot_frame, pack_toolbar=False)
        self.bo_toolbar.update()
        self.bo_toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Info panel
        info_frame = tk.Frame(parent, bg='#ecf0f1', padx=10, pady=10)
        info_frame.pack(fill=tk.X, padx=10, pady=(0,10))
        
        # Best point info
        best_frame = tk.Frame(info_frame, bg='#ecf0f1')
        best_frame.pack(fill=tk.X, pady=(0,5))
        tk.Label(best_frame, text="Best Point Index:", bg='#ecf0f1', font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        self.best_idx_info = tk.StringVar()
        self.best_idx_info.set("None")
        tk.Label(best_frame, textvariable=self.best_idx_info, bg='#ecf0f1', font=('Courier', 9)).pack(side=tk.LEFT, padx=5)
        
        # Score info
        score_frame = tk.Frame(info_frame, bg='#ecf0f1')
        score_frame.pack(fill=tk.X, pady=(0,5))
        tk.Label(score_frame, text="Best Combined Score:", bg='#ecf0f1', font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        self.score_info = tk.StringVar()
        self.score_info.set("None")
        tk.Label(score_frame, textvariable=self.score_info, bg='#ecf0f1', font=('Courier', 9)).pack(side=tk.LEFT, padx=5)
        
        # PCA info
        pca_frame = tk.Frame(info_frame, bg='#ecf0f1')
        pca_frame.pack(fill=tk.X, pady=(0,5))
        tk.Label(pca_frame, text="PCA Explained Variance:", bg='#ecf0f1', font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        self.pca_info = tk.StringVar()
        self.pca_info.set("None")
        tk.Label(pca_frame, textvariable=self.pca_info, bg='#ecf0f1', font=('Courier', 9)).pack(side=tk.LEFT, padx=5)
        
        # Trace dataset changes
        self.bo_dataset_var.trace_add("write", self.update_bo_dataset)
    
    def update_bo_dataset(self, *args):
        """Update BO when dataset is changed"""
        dataset_name = self.bo_dataset_var.get()
        if dataset_name and dataset_name in self.stats_datasets and self.stats_datasets[dataset_name] is not None:
            self.bo_data = self.stats_datasets[dataset_name]
            self.update_bo_column_dropdowns()
            
            # Set default selections if possible
            if len(self.bo_data.columns) >= 5:
                self.input_var.set(", ".join(self.bo_data.columns[:5]))
            if len(self.bo_data.columns) >= 8:
                self.output_var.set(", ".join(self.bo_data.columns[5:8]))
            
            point_count = len(self.bo_data)
            self.bo_status.set(f"Selected: {dataset_name} | Points: {point_count}")
            self.bo_process_btn.config(state=tk.NORMAL)
            self.anim_btn.config(state=tk.NORMAL)
            self.reset_btn.config(state=tk.NORMAL)
        else:
            self.bo_status.set("Please select a valid dataset")
    
    def update_bo_column_dropdowns(self):
        # Clear current dropdown menus
        self.input_dropdown['menu'].delete(0, 'end')
        self.output_dropdown['menu'].delete(0, 'end')
        
        # Add column groups
        if self.bo_data is not None:
            all_cols = self.bo_data.columns.tolist()
            
            # Add all columns as options
            self.input_dropdown['menu'].add_command(
                label="All Columns",
                command=tk._setit(self.input_var, ", ".join(all_cols))
            )
            
            # Add first 5 columns as default option
            if len(all_cols) >= 5:
                self.input_dropdown['menu'].add_command(
                    label="First 5 Columns",
                    command=tk._setit(self.input_var, ", ".join(all_cols[:5]))
                )
            
            # Add all columns as options for outputs
            self.output_dropdown['menu'].add_command(
                label="All Columns",
                command=tk._setit(self.output_var, ", ".join(all_cols))
            )
            
            # Add next 3 columns as default option
            if len(all_cols) >= 8:
                self.output_dropdown['menu'].add_command(
                    label="Columns 6-8",
                    command=tk._setit(self.output_var, ", ".join(all_cols[5:8]))
                )
    
    def visualize_bo(self):
        if self.bo_data is None or not self.input_var.get() or not self.output_var.get():
            self.bo_status.set("Please select input and output columns")
            return
            
        try:
            # Clear previous visualization elements
            self.reset_bo_view()
            
            # Get selected columns
            input_cols = [col.strip() for col in self.input_var.get().split(",")]
            output_cols = [col.strip() for col in self.output_var.get().split(",")]
            
            if len(input_cols) < 2:
                messagebox.showerror("Error", "Please select at least 2 input columns")
                return
            if len(output_cols) < 1:
                messagebox.showerror("Error", "Please select at least 1 output column")
                return
            
            # Get weights
            weights_str = self.weights_var.get().strip()
            if weights_str:
                weights = [float(w.strip()) for w in weights_str.split(",") if w.strip()]
                if len(weights) != len(output_cols):
                    messagebox.showerror("Error", f"Number of weights must match number of output columns ({len(output_cols)})")
                    return
            else:
                # Default to equal weights
                weights = [1.0] * len(output_cols)
            
            # Create a copy to avoid modifying original data
            data_subset = self.bo_data[input_cols + output_cols].copy()
            
            # Convert all columns to numeric, coercing errors
            for col in input_cols + output_cols:
                data_subset[col] = pd.to_numeric(data_subset[col], errors='coerce')
            
            # Drop rows with NaN
            data_subset = data_subset.dropna()
            
            if len(data_subset) == 0:
                raise ValueError("No numeric data available after conversion")
            
            # Extract data
            X = data_subset[input_cols].values
            Y = data_subset[output_cols].values
            
            # Normalize each output column (MinMax scaling to [0,1])
            scaler = MinMaxScaler()
            Y_normalized = scaler.fit_transform(Y)
            
            # Combine normalized outputs into a single score
            self.score = np.dot(Y_normalized, weights)
            
            # Apply PCA to inputs
            self.pca = PCA(n_components=2)
            
            # Handle cases where PCA might fail
            try:
                self.X_pca = self.pca.fit_transform(X)
            except Exception as e:
                # If PCA fails (e.g., constant columns), use standardization
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                self.X_pca = self.pca.fit_transform(X_scaled)
            
            # Find best point
            self.best_idx = np.argmax(self.score)
            
            # Create visualization
            self.create_bo_visualization()
            
            # Update info
            self.best_idx_info.set(str(self.best_idx))
            max_possible_score = np.sum(weights)  # Calculate max possible score
            self.score_info.set(f"{self.score[self.best_idx]:.4f} (max possible: {max_possible_score:.4f})")
            var_ratio = self.pca.explained_variance_ratio_
            self.pca_info.set(f"PC1: {var_ratio[0]*100:.1f}%, PC2: {var_ratio[1]*100:.1f}%")
            self.bo_status.set(f"Visualized BO with {len(X)} points")
            
            # Enable save animation button
            self.save_anim_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            messagebox.showerror("Error", f"BO visualization failed:\n{str(e)}\n\nDetails:\n{error_details}")
    
    def create_bo_visualization(self):
        # Create consistent colormap
        colors = [
            "#ffffcc", "#ffeda0", "#fed976", "#feb24c", 
            "#fd8d3c", "#fc4e2a", "#e31a1c", "#bd0026", "#800026"
        ]
        self.cmap = LinearSegmentedColormap.from_list("bo_cmap", colors)
        
        # Normalize score for coloring - handle case where all scores are identical
        if self.score.max() - self.score.min() > 1e-6:
            norm_score = (self.score - self.score.min()) / (self.score.max() - self.score.min())
        else:
            # If all scores are identical, set all to 0.5 (midpoint)
            norm_score = np.full_like(self.score, 0.5)
        
        # Scatter plot with color mapping
        sc = self.bo_ax.scatter(
            self.X_pca[:, 0], self.X_pca[:, 1],
            c=norm_score, cmap=self.cmap, s=80,
            edgecolor='k', alpha=0.8
        )
        
        # Add colorbar
        self.cbar = self.bo_fig.colorbar(sc, ax=self.bo_ax)
        self.cbar.set_label('Combined Output Score (darker = better)', fontsize=10)
        
        # Add arrows to show BO progression
        for i in range(1, len(self.X_pca)):
            self.bo_ax.annotate("",
                xy=self.X_pca[i],
                xytext=self.X_pca[i-1],
                arrowprops=dict(arrowstyle="->", color='gray', alpha=0.5, lw=1)
            )
        
        # Calculate offset based on axis range
        x_range = self.bo_ax.get_xlim()[1] - self.bo_ax.get_xlim()[0]
        y_range = self.bo_ax.get_ylim()[1] - self.bo_ax.get_ylim()[0]
        offset_x = x_range * self.star_offset_x
        offset_y = y_range * self.star_offset_y
        
        # Highlight best point with star and offset
        self.bo_ax.scatter(
            self.X_pca[self.best_idx, 0] + offset_x, 
            self.X_pca[self.best_idx, 1] + offset_y,
            s=400, marker='*', color='gold', edgecolor='k', linewidth=2,
            zorder=10, label='Best Point'
        )
        
        # Add label for best point
        self.bo_ax.annotate(
            f"Best: {self.score[self.best_idx]:.4f}",
            xy=(self.X_pca[self.best_idx, 0] + offset_x, 
                self.X_pca[self.best_idx, 1] + offset_y),
            xytext=(self.X_pca[self.best_idx, 0] + offset_x * 1.5, 
                   self.X_pca[self.best_idx, 1] + offset_y * 1.5),
            arrowprops=dict(arrowstyle="->", color='black'),
            bbox=dict(boxstyle="round", fc="w", alpha=0.8),
            fontsize=10
        )
        
        # Add legend
        best_patch = mpatches.Patch(color='gold', label='Best Point')
        self.bo_ax.legend(handles=[best_patch], loc='best')
        
        # Add labels and title
        self.bo_ax.set_title("Bayesian Optimization Visualization", fontsize=16)
        self.bo_ax.set_xlabel("PC1 (Summary of Inputs)", fontsize=12)
        self.bo_ax.set_ylabel("PC2 (Another Summary)", fontsize=12)
        self.bo_ax.grid(True, alpha=0.3)
        
        # Set consistent axis limits
        x_min, x_max = self.X_pca[:, 0].min(), self.X_pca[:, 0].max()
        y_min, y_max = self.X_pca[:, 1].min(), self.X_pca[:, 1].max()
        x_range = x_max - x_min
        y_range = y_max - y_min
        self.bo_ax.set_xlim(x_min - 0.1*x_range, x_max + 0.1*x_range)
        self.bo_ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
        
        # Apply tight layout to prevent shifting
        self.bo_fig.tight_layout()
        self.bo_canvas.draw()
    
    def apply_star_offset(self):
        """Apply the star offset to the best point annotation"""
        if self.X_pca is None or self.best_idx is None:
            return
            
        # Get offset values
        self.star_offset_x = self.star_offset_x_var.get()
        self.star_offset_y = self.star_offset_y_var.get()
        
        # Recreate the visualization with the new offset
        self.create_bo_visualization()
    
    def reset_bo_view(self):
        """Reset the BO plot to its initial state"""
        # Clear everything
        self.bo_ax.clear()
        self.bo_ax.grid(True, alpha=0.3)
        self.bo_ax.set_title("Bayesian Optimization Visualization", fontsize=16)
        self.bo_ax.set_xlabel("PC1 (Summary of Inputs)", fontsize=12)
        self.bo_ax.set_ylabel("PC2 (Another Summary)", fontsize=12)
        
        # Remove colorbar if exists
        if hasattr(self, 'cbar') and self.cbar:
            try:
                self.cbar.remove()
            except:
                pass
            self.cbar = None
            
        # Reset data
        self.X_pca = None
        self.score = None
        self.best_idx = None
        
        self.bo_canvas.draw()
        self.bo_status.set("Plot reset to initial state")
    
    def toggle_animation(self):
        """Start or stop the BO animation"""
        if self.animation_running:
            # Stop animation
            if self.bo_animation:
                self.bo_animation.event_source.stop()
                self.bo_animation = None
            self.animation_running = False
            self.anim_btn.config(text="Start Animation", bg='#3498db')
            self.bo_status.set("Animation stopped")
        else:
            # Start animation
            self.animation_running = True
            self.anim_btn.config(text="Stop Animation", bg='#e74c3c')
            self.animate_bo_progression()
    
    def animate_bo_progression(self):
        """Animate the BO process point by point"""
        if self.X_pca is None or self.score is None:
            self.bo_status.set("No data for animation")
            return
            
        # Clear previous animation if exists
        if self.bo_animation:
            self.bo_animation.event_source.stop()
            self.bo_animation = None
        
        # Setup figure
        self.bo_ax.clear()
        self.bo_ax.grid(True, alpha=0.3)
        self.bo_ax.set_title("Bayesian Optimization Progression", fontsize=16)
        self.bo_ax.set_xlabel("PC1 (Summary of Inputs)", fontsize=12)
        self.bo_ax.set_ylabel("PC2 (Another Summary)", fontsize=12)
        
        # Set axis limits based on full data
        x_min, x_max = self.X_pca[:, 0].min(), self.X_pca[:, 0].max()
        y_min, y_max = self.X_pca[:, 1].min(), self.X_pca[:, 1].max()
        x_range = x_max - x_min
        y_range = y_max - y_min
        self.bo_ax.set_xlim(x_min - 0.1*x_range, x_max + 0.1*x_range)
        self.bo_ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
        
        # Create consistent colormap (same as static plot)
        colors = [
            "#ffffcc", "#ffeda0", "#fed976", "#feb24c", 
            "#fd8d3c", "#fc4e2a", "#e31a1c", "#bd0026", "#800026"
        ]
        cmap = LinearSegmentedColormap.from_list("bo_cmap", colors)
        
        # Pre-calculate normalized scores for coloring
        if self.score.max() - self.score.min() > 1e-6:
            self.norm_scores = (self.score - self.score.min()) / (self.score.max() - self.score.min())
        else:
            # If all scores are identical, set all to 0.5 (midpoint)
            self.norm_scores = np.full_like(self.score, 0.5)
        
        # Initialize plot elements with consistent colormap
        scat = self.bo_ax.scatter([], [], s=80, cmap=cmap, edgecolor='k', alpha=0.8)
        best_point = self.bo_ax.scatter([], [], s=400, marker='*', color='gold', edgecolor='k', zorder=10)
        arrows = []
        
        # Calculate max possible score
        weights_str = self.weights_var.get().strip()
        if weights_str:
            weights = [float(w.strip()) for w in weights_str.split(",") if w.strip()]
            max_possible = np.sum(weights)
        else:
            max_possible = len(self.weights_var.get().split(","))
        
        def update(frame):
            """Update function for animation"""
            try:
                n_points = frame + 1
                
                # Update scatter plot
                x_data = self.X_pca[:n_points, 0]
                y_data = self.X_pca[:n_points, 1]
                colors = self.norm_scores[:n_points]  # Use pre-calculated normalized scores
                
                scat.set_offsets(np.c_[x_data, y_data])
                scat.set_array(colors)
                scat.set_clim(0, 1)   # Set color limits to [0,1]
                
                # Update best point
                best_idx = np.argmax(self.score[:n_points])
                best_point.set_offsets([self.X_pca[best_idx, 0], self.X_pca[best_idx, 1]])
                
                # Update arrows
                for arrow in arrows:
                    arrow.remove()
                arrows.clear()
                
                for i in range(1, n_points):
                    arrow = self.bo_ax.annotate("",
                        xy=self.X_pca[i],
                        xytext=self.X_pca[i-1],
                        arrowprops=dict(arrowstyle="->", color='gray', alpha=0.5, lw=1)
                    )
                    arrows.append(arrow)
                
                # Update status
                self.bo_status.set(f"Point {n_points}/{len(self.X_pca)} | Best: {self.score[best_idx]:.4f} (max possible: {max_possible:.4f})")
                
                return scat, best_point, *arrows
            except Exception as e:
                print(f"Animation update error: {e}")
                return scat, best_point
        
        # Get animation interval based on current speed setting
        base_interval = 500 - (400 * self.speed_var.get())
        interval = max(50, min(500, base_interval))  # Clamp between 50-500ms
        
        # Create animation
        self.bo_animation = FuncAnimation(
            self.bo_fig, 
            update, 
            frames=len(self.X_pca),
            interval=interval,
            blit=True,
            repeat=False
        )
        
        self.bo_canvas.draw()
    
    def save_animation(self):
        """Save the BO animation as a video file"""
        if not self.bo_animation:
            messagebox.showwarning("Warning", "No animation to save. Please create an animation first.")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Animation As",
            filetypes=[
                ("MP4 files", "*.mp4"),
                ("GIF files", "*.gif"),
                ("All files", "*.*")
            ],
            defaultextension=".mp4"
        )
        
        if not file_path:
            return
            
        try:
            # Calculate frame interval based on current speed
            base_interval = 500 - (400 * self.speed_var.get())
            interval = max(50, min(500, base_interval))  # Clamp between 50-500ms
            
            # Calculate fps based on interval
            fps = 1000 / interval
            
            # Get the writer based on file extension
            if file_path.lower().endswith('.gif'):
                writer = 'pillow'
                extra_args = {'fps': fps}
            else:
                writer = FFMpegWriter(fps=fps, bitrate=1800)
                extra_args = {}
            
            # Save the animation
            self.bo_animation.save(file_path, writer=writer, **extra_args)
            self.bo_status.set(f"Animation saved to: {file_path}")
            messagebox.showinfo("Success", f"Animation saved successfully to:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save animation:\n{str(e)}")
    
    def create_data_selection_tab(self, parent):
        """Set up the Data Selection tab for violin plots and statistics"""
        # Header
        header = ttk.Label(
            parent, 
            text="Data Selection for All Tabs", 
            font=("Arial", 16, "bold")
        )
        header.pack(pady=20)
        
        # Instructions
        instructions = ttk.Label(
            parent,
            text="1. Import a CSV file containing experimental data\n"
                 "2. Define datasets by selecting rows for each dataset\n"
                 "3. Use the datasets in other tabs for analysis",
            justify="left"
        )
        instructions.pack(pady=10)
        
        # File selection
        file_frame = ttk.Frame(parent)
        file_frame.pack(pady=20, fill="x", padx=50)
        
        self.stats_file_label = ttk.Label(file_frame, text="No file selected")
        self.stats_file_label.pack(side="left", padx=10)
        
        browse_btn = ttk.Button(
            file_frame, 
            text="Browse CSV", 
            command=self.load_stats_csv
        )
        browse_btn.pack(side="right", padx=10)
        
        # Preview area
        preview_label = ttk.Label(parent, text="Data Preview:")
        preview_label.pack(pady=(30, 5), anchor="w", padx=50)
        
        self.stats_preview_text = scrolledtext.ScrolledText(
            parent, 
            height=15,
            state="disabled"
        )
        self.stats_preview_text.pack(fill="both", expand=True, padx=50, pady=(0, 20))
        
        # Dataset selection buttons
        button_frame = ttk.Frame(parent)
        button_frame.pack(pady=20, fill="x", padx=50)
        
        self.dataset_buttons = {}
        for i, ds_name in enumerate(["Dataset 1", "Dataset 2", "Dataset 3", "Dataset 4"]):
            btn = ttk.Button(
                button_frame, 
                text=f"Select {ds_name}", 
                command=lambda n=ds_name: self.open_selection_dialog(n),
                width=20
            )
            btn.grid(row=i//2, column=i%2, padx=10, pady=10, sticky="ew")
            self.dataset_buttons[ds_name] = btn
            btn.config(state="disabled")
            
        # Total dataset button
        total_btn = ttk.Button(
            parent, 
            text="Generate Total Dataset", 
            command=self.generate_total_dataset,
            width=30
        )
        total_btn.pack(pady=20)
        
        # Status labels
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill="x", padx=50, pady=20)
        
        self.status_labels = {}
        for ds_name in ["Dataset 1", "Dataset 2", "Dataset 3", "Dataset 4", "Total"]:
            frame = ttk.Frame(status_frame)
            frame.pack(fill="x", pady=5)
            
            label = ttk.Label(frame, text=f"{ds_name}:", width=12)
            label.pack(side="left")
            
            status = ttk.Label(frame, text="Not selected", foreground="red")
            status.pack(side="left", padx=10)
            self.status_labels[ds_name] = status
        
        # Apply button
        apply_btn = ttk.Button(
            parent, 
            text="Apply Selections", 
            command=self.apply_selections
        )
        apply_btn.pack(pady=20)
    
    def load_stats_csv(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            # Read entire file content
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                content = f.readlines()
            
            # Find header row
            header_line = None
            for i, line in enumerate(content):
                if "GradientTime" in line and "InitalOM" in line:
                    header_line = i
                    break
            
            if header_line is None:
                raise ValueError("Header row not found in CSV file")
            
            # Extract column names from header
            headers = [col.strip() for col in content[header_line].split(',')]
            
            # Process data rows
            data_rows = []
            current_dataset = []
            
            for line in content[header_line+1:]:
                stripped = line.strip()
                if not stripped:  # Skip empty lines
                    continue
                    
                # Check if new dataset header
                if "GradientTime" in stripped and "InitalOM" in stripped:
                    if current_dataset:  # Save current dataset
                        data_rows.extend(current_dataset)
                        current_dataset = []
                    continue
                    
                current_dataset.append(line)
            
            # Add last dataset
            if current_dataset:
                data_rows.extend(current_dataset)
            
            # Create DataFrame
            self.stats_data = pd.read_csv(
                io.StringIO(''.join(data_rows)), 
                names=headers,
                skipinitialspace=True,
                engine='python'
            )
            
            # Clean numeric columns
            self.clean_numeric_columns()
            
            # Update UI
            self.stats_file_label.config(text=os.path.basename(file_path))
            self.show_stats_data_preview()
            
            # Enable dataset buttons
            for btn in self.dataset_buttons.values():
                btn["state"] = "normal"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Could not load file:\n{str(e)}")
    
    def clean_numeric_columns(self):
        """Clean and convert numeric columns, handling various formats"""
        if self.stats_data is None:
            return
            
        for col in self.stats_data.columns:
            try:
                # Skip if column appears numeric already
                if pd.api.types.is_numeric_dtype(self.stats_data[col]):
                    continue
                    
                # Create a working copy of the column
                cleaned_col = self.stats_data[col].astype(str).str.strip()
                
                # Replace empty strings with NaN
                cleaned_col = cleaned_col.replace('', np.nan)
                
                # Handle percentage signs
                if cleaned_col.str.endswith('%').any():
                    cleaned_col = cleaned_col.str.replace('%', '')
                    self.stats_data[col] = pd.to_numeric(cleaned_col, errors='coerce') / 100.0
                else:
                    # Handle commas and other non-numeric characters
                    cleaned_col = cleaned_col.str.replace(',', '')
                    self.stats_data[col] = pd.to_numeric(cleaned_col, errors='coerce')
                    
                # Convert integers to floats if needed
                if self.stats_data[col].dtype == 'int64':
                    self.stats_data[col] = self.stats_data[col].astype(float)
                    
            except Exception as e:
                print(f"Error cleaning column {col}: {str(e)}")
                # If conversion fails, leave the column as-is
                continue
    
    def show_stats_data_preview(self):
        self.stats_preview_text.config(state="normal")
        self.stats_preview_text.delete(1.0, tk.END)
        
        if self.stats_data is not None:
            self.stats_preview_text.insert(tk.END, f"Data Preview (First 10 rows):\n")
            
            # Display with column headers
            if not self.stats_data.empty:
                # Create a display copy without index
                display_df = self.stats_data.head(10).copy()
                display_df.reset_index(drop=True, inplace=True)
                self.stats_preview_text.insert(tk.END, display_df.to_string(index=False) + "\n")
            else:
                self.stats_preview_text.insert(tk.END, "No data available\n")
                
            self.stats_preview_text.insert(tk.END, "\n" + "="*80 + "\n")
            self.stats_preview_text.insert(tk.END, f"Data Info:\nRows极 {len(self.stats_data)}, Columns: {len(self.stats_data.columns)}\n")
            self.stats_preview_text.insert(tk.END, f"Column names: {', '.join(self.stats_data.columns)}")
            
            # Check for potential numeric columns
            numeric_cols = []
            for col in self.stats_data.columns:
                if self.is_column_numeric(self.stats_data[col]):
                    numeric_cols.append(col)
            
            if numeric_cols:
                self.stats_preview_text.insert(tk.END, f"\n\nPotential numeric columns: {', '.join(numeric_cols)}")
            else:
                self.stats_preview_text.insert(tk.END, "\极\nWarning: No numeric columns detected. Plots may not work.")
        
        self.stats_preview_text.config(state="disabled")
    
    def is_column_numeric(self, column):
        """Check if a column contains numeric data, ignoring non-numeric values"""
        try:
            # Attempt to convert to numeric
            cleaned_col = column.astype(str).str.replace(',', '').str.replace('%', '')
            numeric_vals = pd.to_numeric(cleaned_col, errors='coerce')
            # Check if we have at least one valid number
            return numeric_vals.notna().any()
        except Exception as e:
            print(f"Error checking numeric column: {str(e)}")
            return False
    
    def open_selection_dialog(self, dataset_name):
        if self.stats_data is None:
            messagebox.showwarning("Warning", "No data loaded. Please import a CSV file first.")
            return
            
        # Open selection dialog
        dialog = SelectionDialog(self.root, self.stats_data, title=f"Select {dataset_name}")
        
        if dialog.result is not None:
            self.stats_datasets[dataset_name] = dialog.result
            row_count = len(self.stats_datasets[dataset_name])
            self.status_labels[dataset_name].config(
                text=f"{row_count} rows selected", 
                foreground="green"
            )
            # Automatically update other tabs
            self.apply_selections()
    
    def generate_total_dataset(self):
        # Combine all datasets into total
        all_dfs = [df for df in self.stats_datasets.values() 
                  if df is not None and df is not self.stats_datasets["Total"]]
        
        if not all_dfs:
            messagebox.showwarning("Warning", "Please select at least one dataset first.")
            return
            
        self.stats_datasets["Total"] = pd.concat(all_dfs, ignore_index=True)
        row_count = len(self.stats_datasets["Total"])
        self.status_labels["Total"].config(
            text=f"{row_count} rows (combined)", 
            foreground="green"
        )
        # Automatically update other tabs
        self.apply_selections()
        messagebox.showinfo("Success", f"Total dataset created with {row_count} rows")
    
    def apply_selections(self):
        # Validate that we have at least one dataset
        if all(df is None for df in self.stats_datasets.values()):
            messagebox.showwarning("Warning", "No datasets selected. Please select at least one dataset.")
            return
            
        # Get numeric variables from all datasets
        self.stats_numeric_vars = []
        for df in self.stats_datasets.values():
            if df is not None:
                for col in df.columns:
                    if self.is_column_numeric(df[col]) and col not in self.stats_numeric_vars:
                        self.stats_numeric_vars.append(col)
        
        # Update dataset dropdowns in other tabs
        self.update_dataset_dropdowns()
        
        # Generate violin plots and statistics
        self.generate_violin_plots()
        self.generate_statistics_tables()
        
        # Update status
        self.status.set("Datasets applied successfully!")
        self.bo_status.set("Datasets applied successfully!")
    
    def update_dataset_dropdowns(self):
        """Update dataset dropdowns in GPR and BO tabs"""
        # Get available datasets
        available_datasets = [name for name, df in self.stats_datasets.items() if df is not None]
        
        # Update GPR tab dropdown
        menu = self.dataset_dropdown['menu']
        menu.delete(0, 'end')
        self.dataset_dropdown.config(state=tk.NORMAL)
        for ds in available_datasets:
            menu.add_command(label=ds, command=tk._setit(self.dataset_var, ds))
        
        # Set default selection if available
        if available_datasets:
            self.dataset_var.set(available_datasets[0])
        
        # Update BO tab dropdown
        menu_bo = self.bo_dataset_dropdown['menu']
        menu_bo.delete(0, 'end')
        self.bo_dataset_dropdown.config(state=tk.NORMAL)
        for ds in available_datasets:
            menu_bo.add_command(label=ds, command=tk._setit(self.bo_dataset_var, ds))
        
        # Set default selection if available
        if available_datasets:
            self.bo_dataset_var.set(available_datasets[0])
    
    def generate_violin_plots(self):
        # Clear previous content
        for widget in self.violin_tab.winfo_children():
            widget.destroy()
        
        if all(df is None for df in self.stats_datasets.values()):
            ttk.Label(self.violin_tab, text="No datasets defined. Please select data first.").pack(pady=50)
            return
        
        if not self.stats_numeric_vars:
            error_frame = ttk.Frame(self.violin_tab)
            error_frame.pack(fill="both", expand=True, padx=50, pady=50)
            
            ttk.Label(error_frame, 
                     text="No numeric variables found in data.\n\n"
                     "Possible reasons:\n"
                     "1. Your data doesn't contain numeric values\n"
                     "2. Numeric columns contain non-numeric characters\n"
                     "3. CSV format issues\n\n"
                     "Please check your data and try again.",
                     justify="left",
                     foreground="red").pack()
            return
            
        # Create control frame at top
        control_frame = ttk.Frame(self.violin_tab)
        control_frame.pack(fill="x", padx=20, pady=10)
        
        # Variable selection
        ttk.Label(control_frame, text="Select Variables:").pack(side="left", padx=5)
        
        self.var_listbox = tk.Listbox(
            control_frame, 
            selectmode=tk.MULTIPLE,
            width=30,
            height=6
        )
        self.var_listbox.pack(side="left", padx=5)
        
        # Populate listbox
        for var in self.stats_numeric_vars:
            self.var_listbox.insert(tk.END, var)
        
        # Select first 4 by default
        for i in range(min(4, len(self.stats_numeric_vars))):
            self.var_listbox.select_set(i)
        
        # Button to generate plot
        plot_btn = ttk.Button(
            control_frame, 
            text="Generate Plots", 
            command=self.generate_selected_violin_plot
        )
        plot_btn.pack(side="left", padx=10)
        
        # Save plot button
        save_btn = ttk.Button(
            control_frame, 
            text="Save Plot", 
            command=self.save_current_violin_plot
        )
        save_btn.pack(side="right", padx=5)
        
        # Create plot container frame
        self.plot_frame = ttk.Frame(self.violin_tab)
        self.plot_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        # Generate initial plot
        self.generate_selected_violin_plot()
    
    def generate_selected_violin_plot(self):
        # Clear previous plot if it exists
        if hasattr(self, 'current_violin_plot_canvas') and self.current_violin_plot_canvas:
            self.current_violin_plot_canvas.get_tk_widget().destroy()
            plt.close(self.current_violin_plot_fig)
        
        # Get selected variables
        selected_vars = [self.var_listbox.get(i) for i in self.var_listbox.curselection()]
        if not selected_vars:
            return
            
        # Create figure with subplots - using tight_layout instead of constrained_layout
        n = len(selected_vars)
        n_cols = min(2, n)  # Max 2 columns
        n_rows = (n + n_cols - 1) // n_cols
        
        # Use larger figure size with tight_layout
        self.current_violin_plot_fig, axes = plt.subplots(
            n_rows, n_cols, 
            figsize=(10, 6 * n_rows),  # Increased height
            squeeze=False
        )
        axes = axes.flatten()
        
        # Track if we have any valid plots
        valid_plots = 0
        
        # Create a plot for each selected variable
        for i, var in enumerate(selected_vars):
            ax = axes[i]
            plot_data = []
            
            for ds_name, df in self.stats_datasets.items():
                if df is not None and var in df.columns and ds_name != "Total":
                    # Convert to numeric, handling errors
                    cleaned_col = df[var].astype(str).str.replace(',', '').str.replace('%', '')
                    numeric_vals = pd.to_numeric(cleaned_col, errors='coerce').dropna()
                    
                    # Skip if no data or all values identical
                    if len(numeric_vals) == 0:
                        continue
                        
                    if numeric_vals.nunique() == 1:
                        # Handle constant values by adding slight variation
                        numeric_vals += np.random.normal(0, 1e-5, len(numeric_vals))
                    
                    for value in numeric_vals:
                        plot_data.append({"Dataset": ds_name, "Value": value})
            
            if not plot_data:
                ax.axis('off')  # Hide empty axes
                continue
                
            plot_df = pd.DataFrame(plot_data)
            valid_plots += 1
            
            # Create violin plot with fixed parameters
            sns.violinplot(
                x="Dataset", 
                y="Value", 
                hue="Dataset",  # Fix for FutureWarning
                data=plot_df, 
                ax=ax,
                palette="viridis",
                inner="quartile",
                cut=0,
                density_norm="width",  # Fix for FutureWarning
                legend=False  # Fix for FutureWarning
            )
            
            # Special formatting
            if "Crit" in var:
                ax.axhline(0, color='red', linestyle='--', alpha=0.5)
            if "Time" in var:
                ax.set_yscale('log')
            
            ax.set_title(f"{var} Distribution", fontsize=12)
            ax.set_xlabel('')
            ax.set_ylabel(var, fontsize=10)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add number of observations - position below x-axis
            n_obs = plot_df.groupby('Dataset').size()
            x_ticks = ax.get_xticks()
            y_min, y_max = ax.get_ylim()
            
            # Position n= labels below the x-axis
            label_y_pos = y_min - 0.2 * (y_max - y_min)
            
            for j, ds in enumerate(ax.get_xticklabels()):
                ds_name = ds.get_text()
                count = n_obs.get(ds_name, 0)
                ax.text(x_ticks[j], label_y_pos, 
                       f'n={count}', 
                       ha='center', va='top', fontsize=9)
        
        # Hide unused axes
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        
        if valid_plots == 0:
            # Show message if no valid plots
            for ax in axes:
                ax.axis('off')
            self.current_violin_plot_fig.text(0.5, 0.5, 
                                             'No numeric data available for selected variables',
                                             ha='center', va='center', fontsize=12)
        else:
            # Apply tight_layout only if we have valid plots
            self.current_violin_plot_fig.tight_layout()
        
        # Embed plot in Tkinter
        self.current_violin_plot_canvas = FigureCanvasTkAgg(self.current_violin_plot_fig, master=self.plot_frame)
        self.current_violin_plot_canvas.draw()
        self.current_violin_plot_canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def save_current_violin_plot(self):
        if not hasattr(self, 'current_violin_plot_fig') or not self.current_violin_plot_fig:
            messagebox.showwarning("Warning", "No plot to save")
            return
            
        selected_vars = [self.var_listbox.get(i) for i in self.var_listbox.curselection()]
        if not selected_vars:
            var = "plot"
        else:
            var = "_".join(selected_vars[:3])
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")],
            initialfile=f"{var}_distribution.png"
        )
        if file_path:
            self.current_violin_plot_fig.savefig(file_path, dpi=300, bbox_inches="tight")
            messagebox.showinfo("Success", f"Plot saved to:\n{file_path}")
    
    def generate_statistics_tables(self):
        # Clear previous content
        for widget in self.stats_tab.winfo_children():
            widget.destroy()
        
        if all(df is None for df in self.stats_datasets.values()):
            # Use grid for consistency
            label = ttk.Label(self.stats_tab, text="No datasets defined. Please select data first.")
            label.grid(row=0, column=0, sticky="nsew", pady=50)
            return
        
        # Create notebook for dataset stats
        stats_notebook = ttk.Notebook(self.stats_tab)
        stats_notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create tab for each dataset
        for ds_name, df in self.stats_datasets.items():
            if df is None:
                continue
                
            tab = ttk.Frame(stats_notebook)
            stats_notebook.add(tab, text=ds_name)
            
            # Create container frame to avoid mixing geometry managers
            container = ttk.Frame(tab)
            container.pack(fill="both", expand=True)
            
            # Get numeric columns
            numeric_cols = []
            for col in df.columns:
                if self.is_column_numeric(df[col]):
                    numeric_cols.append(col)
            
            # Skip if no numeric columns
            if not numeric_cols:
                # Create a frame for the error message
                error_frame = ttk.Frame(container)
                error_frame.pack(fill="both", expand=True, padx=50, pady=50)
                
                ttk.Label(error_frame, 
                         text=f"No numeric columns found in {ds_name}\n"
                         "Possible reasons:\n"
                         "1. The dataset doesn't contain numeric values\n"
                         "2. Numeric columns contain non-numeric characters\n"
                         "3. The dataset is empty",
                         justify="left",
                         foreground="red").pack()
                continue
            
            # Create a numeric-only version of the dataset for statistics
            numeric_df = df[numeric_cols].apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '').str.replace('%', ''), errors='coerce'))
            
            # Create table
            tree = ttk.Treeview(container, show="headings")
            vsb = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
            hsb = ttk.Scrollbar(container, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
            
            # Layout using grid
            tree.grid(row=0, column=0, sticky="nsew")
            vsb.grid(row=0, column=1, sticky="ns")
            hsb.grid(row=1, column=0, sticky="ew")
            
            # Configure grid weights
            container.grid_rowconfigure(0, weight=1)
            container.grid_columnconfigure(0, weight=1)
            
            # Setup columns
            tree["columns"] = ["Statistic"] + numeric_cols
            tree.column("#0", width=0, stretch=tk.NO)  # Hide first column
            
            # Create headings
            tree.heading("Statistic", text="Statistic")
            for col in numeric_cols:
                tree.heading(col, text=col)
            
            # Calculate statistics using the converted numeric data
            stats_data = {
                "Count": numeric_df.count(),
                "Min": numeric_df.min(),
                "Q1 (25%)": numeric_df.quantile(0.25),
                "Median": numeric_df.median(),
                "Q3 (75%)": numeric_df.quantile(0.75),
                "Max": numeric_df.max(),
                "Mean": numeric_df.mean(),
                "Std Dev": numeric_df.std()
            }
            
            # Add data to table
            for stat_name, values in stats_data.items():
                value_list = [stat_name] + list(values.round(4))
                tree.insert("", "end", values=value_list)
            
            # Adjust column widths
            for col in tree["columns"]:
                tree.column(col, width=100, anchor="center", minwidth=80)
        
        # Create new tab for statistical tests
        stats_test_tab = ttk.Frame(stats_notebook)
        stats_notebook.add(stats_test_tab, text="Statistical Tests")
        
        # Create container for statistical tests
        test_container = ttk.Frame(stats_test_tab)
        test_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Variable selection
        ttk.Label(test_container, text="Select Variable:").pack(pady=5)
        self.test_var_selector = ttk.Combobox(
            test_container, 
            values=self.stats_numeric_vars,
            state="readonly",
            width=30
        )
        self.test_var_selector.pack(pady=5)
        if self.stats_numeric_vars:
            self.test_var_selector.current(0)
        
        # Run tests button
        ttk.Button(
            test_container, 
            text="Run Statistical Tests", 
            command=self.run_statistical_tests
        ).pack(pady=10)
        
        # Results area
        self.test_results_text = scrolledtext.ScrolledText(
            test_container, 
            height=15,
            state="disabled"
        )
        self.test_results_text.pack(fill="both", expand=True, pady=10)
        
        # Create new tab for cosine similarity
        cosine_tab = ttk.Frame(stats_notebook)
        stats_notebook.add(cosine_tab, text="Cosine Similarity")
        
        # Create container for cosine similarity
        cosine_container = ttk.Frame(cosine_tab)
        cosine_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Run analysis button
        ttk.Button(
            cosine_container, 
            text="Calculate Cosine Similarity", 
            command=self.calculate_cosine_similarity
        ).pack(pady=10)
        
        # Results area
        self.cosine_results_text = scrolledtext.ScrolledText(
            cosine_container, 
            height=15,
            state="disabled"
        )
        self.cosine_results_text.pack(fill="both", expand=True, pady=10)
    
    def run_statistical_tests(self):
        try:
            # Create a table of H-test results with pairwise comparisons
            result_text = "Kruskal-Wallis Test with Post-Hoc Analysis\n"
            result_text += "="*90 + "\n"
            result_text += f"{'Variable':<20} {'H-statistic':<12} {'p-value':<12} {'Sig':<6} {'Pairwise Differences':<40}\n"
            result_text += "-"*90 + "\n"
            
            # Prepare data for all variables
            all_vars = self.stats_numeric_vars
            if not all_vars:
                self.show_test_results("No numeric variables available for testing")
                return
                
            # Run tests for all variables
            for var in all_vars:
                # Prepare data groups
                data_groups = []
                dataset_names = []
                
                for ds_name in ["Dataset 1", "Dataset 2", "Dataset 3", "Dataset 4"]:
                    df = self.stats_datasets.get(ds_name)
                    if df is not None and var in df.columns:
                        # Clean and get numeric values
                        cleaned_col = df[var].astype(str).str.replace(',', '').str.replace('%', '')
                        numeric_vals = pd.to_numeric(cleaned_col, errors='coerce').dropna()
                        if len(numeric_vals) > 0:
                            data_groups.append(numeric_vals)
                            dataset_names.append(ds_name)
                
                if len(data_groups) < 2:
                    # Not enough datasets for comparison
                    result_text += f"{var:<20} {'-':<12} {'-':<12} {'-':<6} {'Insufficient data':<40}\n"
                    continue
                    
                try:
                    # Kruskal-Wallis H-test
                    h_stat, p_value = stats.kruskal(*data_groups)
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "NS"
                    
                    # Post-hoc pairwise comparisons with Mann-Whitney U test
                    pairwise_results = []
                    if len(data_groups) > 2 and p_value < 0.05:
                        # Calculate number of pairwise comparisons
                        n_comparisons = len(dataset_names) * (len(dataset_names) - 1) // 2
                        bonferroni_threshold = 0.05 / n_comparisons
                        
                        pairs = []
                        for i in range(len(dataset_names)):
                            for j in range(i+1, len(dataset_names)):
                                # Mann-Whitney U test for pairwise comparison
                                u_stat, p_val = stats.mannwhitneyu(data_groups[i], data_groups[j])
                                if p_val < bonferroni_threshold:
                                    pairs.append(f"{dataset_names[i]}-{dataset_names[j]}:{p_val:.3f}")
                        
                        pairwise_str = ", ".join(pairs) if pairs else "No sig. pairs"
                    else:
                        pairwise_str = "-"
                    
                    # Format results
                    result_text += f"{var:<20} {h_stat:.4f}{'':<8} {p_value:.6f}{'':<6} {significance:<6} {pairwise_str[:40]:<40}\n"
                    
                except Exception as e:
                    result_text += f"{var:<20} {'Error':<12} {'Error':<12} {'-':<6} {'Error in calculation':<40}\n"
                    print(f"Error processing {var}: {str(e)}")
            
            # Add interpretation note
            result_text += "\n" + "="*90 + "\n"
            result_text += "Significance levels: *** p<0.001, ** p<0.01, * p<0.05, NS = Not significant\n"
            result_text += "NOT SIGNIFICANT (NS) means distributions are SIMILAR (no statistical difference)\n"
            result_text += "SIGNIFICANT (*) means distributions are DIFFERENT\n"
            result_text += "Pairwise Differences: Show significantly different pairs with p-values\n"
            result_text += "Note: Analysis uses Bonferroni correction for multiple comparisons"
            
            self.show_test_results(result_text)
            
        except Exception as e:
            self.show_test_results(f"Error performing statistical tests:\n{str(e)}")
    
    def show_test_results(self, text):
        self.test_results_text.config(state="normal")
        self.test_results_text.delete(1.0, tk.END)
        self.test_results_text.insert(tk.END, text)
        self.test_results_text.config(state="disabled")
    
    def calculate_cosine_similarity(self):
        try:
            # Get valid datasets
            valid_datasets = [ds_name for ds_name in ["Dataset 1", "Dataset 2", "Dataset 3", "Dataset 4"] 
                             if self.stats_datasets.get(ds_name) is not None 
                             and not self.stats_datasets[ds_name].empty]
            pairs = []
            for i in range(len(valid_datasets)):
             for j in range(i+1, len(valid_datasets)):
                pairs.append((valid_datasets[i], valid_datasets[j]))
            
            if len(valid_datasets) < 2:
                self.show_cosine_results("Need at least 2 datasets with numeric data")
                return

            # Find common numeric columns across all datasets
            common_columns = set.intersection(*[set(self.stats_datasets[ds_name].columns) 
                                              for ds_name in valid_datasets])
            
            # Filter for numeric columns
            common_numeric_columns = [
                col for col in common_columns 
                if all(self.is_column_numeric(self.stats_datasets[ds_name][col]) for ds_name in valid_datasets)
            ]

            if not common_numeric_columns:
                self.show_cosine_results("No common numeric columns across datasets")
                return

            # Initialize result text
            result_text = "Cosine Similarity Analysis\n"
            result_text += "=" * 70 + "\n"
            result_text += "Values range from -1 (completely dissimilar) to 1 (identical)\n\n"
            
            # Format results as pairs per variable for better readability
            for var in common_numeric_columns:
                result_text += f"\n{var}:\n"
                result_text += "-" * (len(var) + 1) + "\n"
                
                # Prepare data for this variable
                var_data = {}
                min_val = np.inf
                max_val = -np.inf
                
                # First pass: get global min/max for consistent bins
                for ds_name in valid_datasets:
                    # Clean and get numeric values
                    cleaned_col = self.stats_datasets[ds_name][var].astype(str).str.replace(',', '').str.replace('%', '')
                    numeric_vals = pd.to_numeric(cleaned_col, errors='coerce').dropna().values
                    
                    if len(numeric_vals) == 0:
                        continue
                    
                    current_min = np.min(numeric_vals)
                    current_max = np.max(numeric_vals)
                    
                    # Update global min/max only if we have valid values
                    if not np.isnan(current_min) and not np.isnan(current_max):
                        min_val = min(min_val, current_min)
                        max_val = max(max_val, current_max)
                
                # Skip if constant value
                if min_val == max_val:
                    result_text += f"   Skipped: constant value ({min_val}) across all datasets\n"
                    continue
                
                # Skip if we couldn't determine valid min/max
                if np.isinf(min_val) or np.isinf(max_val):
                    result_text += "No valid data for this variable\n"
                    continue
                
                # Create consistent bins across all datasets
                num_bins = 20
                bins = np.linspace(min_val, max_val, num_bins + 1)
                
                # Second pass: create histograms
                for ds_name in valid_datasets:
                    # Clean and get numeric values
                    cleaned_col = self.stats_datasets[ds_name][var].astype(str).str.replace(',', '').str.replace('%', '')
                    numeric_vals = pd.to_numeric(cleaned_col, errors='coerce').dropna().values
                    
                    if len(numeric_vals) == 0:
                        continue
                    
                    # Calculate histogram and ensure no NaN values
                    hist, _ = np.histogram(numeric_vals, bins=bins, density=True)
                    hist = np.nan_to_num(hist)  # Replace any NaN with 0
                    var_data[ds_name] = hist
                
                # Calculate cosine similarity for each pair
                for pair in pairs:
                    ds1, ds2 = pair
                    
                    # Skip if data not available for either dataset
                    if ds1 not in var_data or ds2 not in var_data:
                        result_text += f"   {ds1} vs {ds2}: N/A\n"
                        continue
                    
                    try:
                        # Calculate cosine similarity with NaN handling
                        similarity = cosine_similarity(
                            [var_data[ds1]], 
                            [var_data[ds2]]
                        )[0][0]
                        
                        # Handle potential NaN results
                        if np.isnan(similarity):
                            result_text += f"   {ds1} vs {ds2}: N/A\n"
                        else:
                            result_text += f"   {ds1} vs {ds2}: {similarity:.4f}\n"
                    except:
                        result_text += f"   {ds1} vs {ds2}: Error\n"
            
            # Add interpretation guide at the end
            result_text += "\n" + "=" * 70 + "\n"
            result_text += "Interpretation Guide:\n"
            result_text += "1.00  : Very similar distributions\n"
            result_text += "0.90-0.99: Strongly similar\n"
            result_text += "0.70-0.89: Moderately similar\n"
            result_text += "0.50-0.69: Weakly similar\n"
            result_text += "0.00-0.49: Not similar\n"
            result_text += "< 0.00  : Dissimilar (opposite patterns)\n"
            
            # Add note about methodology
            result_text += "\nMethodology:\n"
            result_text += "- For each variable, created normalized histograms with 20 bins\n"
            result_text += "- Used consistent bin ranges across all datasets\n"
            result_text += "- Calculated cosine similarity between histogram vectors\n"
            
            self.show_cosine_results(result_text)
            
        except Exception as e:
            import traceback
            self.show_cosine_results(f"Error calculating cosine similarity:\n{str(e)}\n\n{traceback.format_exc()}")
    
    def show_cosine_results(self, text):
        self.cosine_results_text.config(state="normal")
        self.cosine_results_text.delete(1.0, tk.END)
        self.cosine_results_text.insert(tk.END, text)
        self.cosine_results_text.config(state="disabled")

if __name__ == "__main__":
    root = tk.Tk()
    app = GPR_GUI(root)
    root.mainloop()