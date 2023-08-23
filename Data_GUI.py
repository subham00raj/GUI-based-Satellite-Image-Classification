import tkinter as tk
from tkinter import simpledialog
import rasterio
import csv
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Define the image file path
image_file = "/Users/subhamraj/Desktop/22103070_Subham_tp/training.tif"

# Open the image using Rasterio
with rasterio.open(image_file) as src:
    # Read the image bands
    band1 = src.read(1)
    band2 = src.read(2)
    band3 = src.read(3)
    band4 = src.read(4)
    # Convert the image data type to uint8
    band1 = (255*((band1-band1.min())/(band1.max()-band1.min()))).astype(np.uint8)
    band2 = (255*((band2-band2.min())/(band2.max()-band2.min()))).astype(np.uint8)
    band3 = (255*((band3-band3.min())/(band3.max()-band3.min()))).astype(np.uint8)
    band4 = (255*((band4-band4.min())/(band4.max()-band4.min()))).astype(np.uint8)
    # Get the image dimensions
    height, width = band1.shape

# Define a dictionary to store the pixel classes and their values
class_values = {}

# Define a function to handle mouse clicks
def on_click(event):
    # Get the pixel coordinates
    x, y = int(event.xdata), int(event.ydata)
    # Get the pixel values
    values = [band1[y, x], band2[y, x], band3[y, x], band4[y, x]]
    # Ask the user for the pixel class
    class_name = simpledialog.askstring("Input", "Enter the pixel class:", parent=root)
    # If the user clicked Cancel, do nothing
    if class_name is None:
        return
    # If the class is not already in the dictionary, add it
    if class_name not in class_values:
        class_values[class_name] = []
    # If the class already has 5 pixels selected, don't add another
    if len(class_values[class_name]) == 5:
        print("Maximum number of pixels reached for class", class_name)
        return
    # Add the values to the class
    class_values[class_name].append(values)
    # Update the attribute table
    with open("output.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Label", "Band_1", "Band_2", "Band_3", "Band_4"])
        for class_name in class_values:
            for values in class_values[class_name]:
                writer.writerow([class_name] + values)

# Create the GUI window
root = tk.Tk()
root.title("Image Viewer")

# Create a Matplotlib figure
fig = Figure(figsize=(width/100, height/100), dpi=100)
ax = fig.add_subplot(111)

# Display the image
ax.imshow(np.dstack((band3, band2, band1)))

# Create a canvas to display the Matplotlib figure
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack()

# Bind the mouse click event to the canvas
cid = fig.canvas.mpl_connect('button_press_event', on_click)

# Start the GUI event loop
root.mainloop()

