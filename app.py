import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import gradio as gr

css = '''
.gradio-container{max-width: 789px !important}
h1{text-align:center}
'''

def create_visualizations(data):
    plots = []
    
    # Create figures directory
    figures_dir = "./figures"
    shutil.rmtree(figures_dir, ignore_errors=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Histograms for numeric columns
    numeric_cols = data.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        plt.figure()
        sns.histplot(data[col], kde=True)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        hist_path = os.path.join(figures_dir, f'histogram_{col}.png')
        plt.savefig(hist_path)
        plt.close()
        plots.append(hist_path)
    
    # Box plots for numeric columns
    for col in numeric_cols:
        plt.figure()
        sns.boxplot(x=data[col])
        plt.title(f'Box Plot of {col}')
        box_path = os.path.join(figures_dir, f'boxplot_{col}.png')
        plt.savefig(box_path)
        plt.close()
        plots.append(box_path)
    
    # Scatter plot matrix
    if len(numeric_cols) > 1:
        plt.figure()
        sns.pairplot(data[numeric_cols])
        plt.title('Scatter Plot Matrix')
        scatter_matrix_path = os.path.join(figures_dir, 'scatter_matrix.png')
        plt.savefig(scatter_matrix_path)
        plt.close()
        plots.append(scatter_matrix_path)
    
    # Correlation heatmap
    if len(numeric_cols) > 1:
        plt.figure()
        corr = data[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        heatmap_path = os.path.join(figures_dir, 'correlation_heatmap.png')
        plt.savefig(heatmap_path)
        plt.close()
        plots.append(heatmap_path)
    
    # Bar charts for categorical columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    if not categorical_cols.empty:
        for col in categorical_cols:
            plt.figure()
            data[col].value_counts().plot(kind='bar')
            plt.title(f'Bar Chart of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            bar_path = os.path.join(figures_dir, f'bar_chart_{col}.png')
            plt.savefig(bar_path)
            plt.close()
            plots.append(bar_path)
    
    # Line charts (if a 'date' column is present)
    if 'date' in data.columns:
        plt.figure()
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date').plot()
        plt.title('Line Chart of Date Series')
        line_chart_path = os.path.join(figures_dir, 'line_chart.png')
        plt.savefig(line_chart_path)
        plt.close()
        plots.append(line_chart_path)
    
    # Scatter plot using Plotly
    if len(numeric_cols) >= 2:
        fig = px.scatter(data, x=numeric_cols[0], y=numeric_cols[1], title='Scatter Plot')
        scatter_plot_path = os.path.join(figures_dir, 'scatter_plot.html')
        fig.write_html(scatter_plot_path)
        plots.append(scatter_plot_path)
    
    # Pie chart for categorical columns (only the first categorical column)
    if not categorical_cols.empty:
        fig = px.pie(data, names=categorical_cols[0], title='Pie Chart of ' + categorical_cols[0])
        pie_chart_path = os.path.join(figures_dir, 'pie_chart.html')
        fig.write_html(pie_chart_path)
        plots.append(pie_chart_path)
    
    # Heatmaps (e.g., for a correlation matrix or cross-tabulation)
    if len(numeric_cols) > 1:
        heatmap_data = data[numeric_cols].corr()
        fig = px.imshow(heatmap_data, text_auto=True, title='Heatmap of Numeric Variables')
        heatmap_plot_path = os.path.join(figures_dir, 'heatmap_plot.html')
        fig.write_html(heatmap_plot_path)
        plots.append(heatmap_plot_path)
    
    # Violin plots for numeric columns
    for col in numeric_cols:
        plt.figure()
        sns.violinplot(x=data[col])
        plt.title(f'Violin Plot of {col}')
        violin_path = os.path.join(figures_dir, f'violin_plot_{col}.png')
        plt.savefig(violin_path)
        plt.close()
        plots.append(violin_path)
    
    return plots

def analyze_data(file_input):
    data = pd.read_csv(file_input.name)
    return create_visualizations(data)

# Example file path
example_file_path = "./example/example.csv"

with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue=gr.themes.colors.red, secondary_hue=gr.themes.colors.blue)) as demo:
    gr.Markdown("# DATA BOARD📊\nUpload a `.csv` file to generate various visualizations and interactive plots.")
    
    file_input = gr.File(label="Upload your `.csv` file")
    submit = gr.Button("Generate Dashboards")
    
    # Display images and interactive plots in a gallery
    gallery = gr.Gallery(label="Visualizations")
    
    # Example block
    examples = gr.Examples(
        examples=[[example_file_path]],
        inputs=file_input,
        outputs=gallery
    )
    
    submit.click(analyze_data, file_input, gallery)

if __name__ == "__main__":
    demo.launch(share=True)