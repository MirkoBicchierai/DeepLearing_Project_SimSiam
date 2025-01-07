import json
import matplotlib.pyplot as plt


"""
Function used to extract values from graphs created with comet_ml, starting from a JSON file, 
and reconstruct the graphs using matplotlib.
"""

def create_graph(path, title, x_label, y_label):

    with open(path, 'r') as f:
        data = json.load(f)

    plt.figure(figsize=(10, 6))

    for trace in data:
        x = trace["x"]
        y = trace["y"]
        name = trace["name"]

        if "NaN" in y:
            y = [float(value) if value != "NaN" else None for value in y]
            x, y = zip(*[(x_val, y_val) for x_val, y_val in zip(x, y) if y_val is not None])

        plt.plot(x, y, label=name)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    file_path = "CometMlData/loss VS step_chart_data.json"
    title_graph = "Loss Vs Epoch"
    y_l = "Loss"
    x_l = "Epochs"
    create_graph(file_path, title_graph, x_l, y_l)







