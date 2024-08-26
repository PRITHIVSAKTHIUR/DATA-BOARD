import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import gradio as gr
from mpl_toolkits.mplot3d import Axes3D

css = '''
.gradio-container{max-width: 900px !important}
h1{text-align:center}
'''

def create_visualizations(data):
    plots = []
    
    figures_dir = "./figures"
    shutil.rmtree(figures_dir, ignore_errors=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    numeric_cols = data.select_dtypes(include=['number']).columns
    data = data.dropna(subset=numeric_cols)
    
    # 3D Histograms for numeric columns
    for col in numeric_cols:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        hist, xedges, yedges = np.histogram2d(data[col], data[col], bins=20)
        xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = 0

        dx = dy = 0.5 * np.ones_like(zpos)
        dz = hist.ravel()

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
        ax.set_title(f'3D Histogram of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel(col)
        ax.set_zlabel('Frequency')
        hist_path = os.path.join(figures_dir, f'3d_histogram_{col}.png')
        plt.savefig(hist_path)
        plt.close()
        plots.append(hist_path)
    
    for col in numeric_cols:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        data_array = np.array([data[col]])
        ax.boxplot(data_array)
        ax.set_title(f'3D Box Plot of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Category')
        ax.set_zlabel('Value')
        box_path = os.path.join(figures_dir, f'3d_boxplot_{col}.png')
        plt.savefig(box_path)
        plt.close()
        plots.append(box_path)
    
    if len(numeric_cols) >= 2:
        for i in range(len(numeric_cols) - 1):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data[numeric_cols[i]], data[numeric_cols[i + 1]], data[numeric_cols[i + 1]])
            ax.set_title(f'3D Scatter Plot of {numeric_cols[i]} vs {numeric_cols[i + 1]}')
            ax.set_xlabel(numeric_cols[i])
            ax.set_ylabel(numeric_cols[i + 1])
            ax.set_zlabel(numeric_cols[i + 1])
            scatter_path = os.path.join(figures_dir, f'3d_scatter_plot_{numeric_cols[i]}_{numeric_cols[i + 1]}.png')
            plt.savefig(scatter_path)
            plt.close()
            plots.append(scatter_path)
    
    if len(numeric_cols) >= 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = data[numeric_cols[0]]
        y = data[numeric_cols[1]]
        z = data[numeric_cols[2]]
        ax.plot_trisurf(x, y, z, cmap='viridis')
        ax.set_title(f'3D Surface Plot')
        ax.set_xlabel(numeric_cols[0])
        ax.set_ylabel(numeric_cols[1])
        ax.set_zlabel(numeric_cols[2])
        surface_path = os.path.join(figures_dir, '3d_surface_plot.png')
        plt.savefig(surface_path)
        plt.close()
        plots.append(surface_path)
    
    return plots

def analyze_data(file_input):
    data = pd.read_csv(file_input.name)
    return create_visualizations(data)

example_file_path = "./example/🤗example.csv"

with gr.Blocks(css=css, theme="bethecloud/storj_theme") as demo:
    gr.Markdown("# DATA BOARD 3D📊\nUpload a `.csv` file to generate various visualizations and interactive plots.")
    
    file_input = gr.File(label="Upload your `.csv` file")
    
    submit = gr.Button("Generate Dashboards")
    
    gallery = gr.Gallery(label="Visualizations")

    examples = gr.Examples(
        examples=[[example_file_path]],
        inputs=file_input,
        outputs=gallery,
        fn=analyze_data,  
        cache_examples=True  
    )
    
    submit.click(analyze_data, file_input, gallery)

if __name__ == "__main__":
    demo.launch()