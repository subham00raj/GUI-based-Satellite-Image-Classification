from tkinter import *
from tkinter import filedialog
from tkinter import colorchooser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

root = Tk()
root.title('Support Vector Machines Image Classifier')
root.geometry('500x600')

csv_path = ''
def select_file():
    global csv_path
    filetype = (('Text files', '.txt'), ('CSV files', '.csv'), ('All files', '.*'))
    csv_path = filedialog.askopenfilename(title='Select a file', filetypes=filetype)
    return csv_path

tif_path = ''
def select_imagefile():
    global tif_path
    filetype = (('TIFF files', '.tif'), ('All files', '.*'))
    tif_path = filedialog.askopenfilename(title='Select a file', filetypes=filetype)
    return tif_path

def readcsv():    
    df = pd.read_csv(csv_path)
    data = df[['Band_1', 'Band_2', 'Band_3']]
    label = df['Label']
    return data, label

def Classifier(kernel):
    data, label = readcsv()
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, stratify=label.ravel(), random_state=42)
    svm_model = svm.SVC(kernel=kernel, C=float(E1.get()),degree = int(E2.get()),gamma=int(E3.get()))
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    y_pred_all = svm_model.predict(raw_image(tif_path))
    return y_test,y_pred,y_pred_all

def raw_image(tif_path):
    with rasterio.open(tif_path) as src:
        array = src.read()
        rows = src.height
        cols = src.width
        bands = src.count
        array = np.moveaxis(array, 0, -1)
        array = np.reshape(array, [rows*cols, bands])
        test = pd.DataFrame(array, dtype='int16', columns=['Band_1', 'Band_2', 'Band_3', 'Band_4'])
        test.drop('Band_4', axis=1, inplace=True)
    return test

def on_color_button(index):
    color = colorchooser.askcolor()
    if color:
        colors[index] = [int(c) for c in color[0]]
        color_buttons[index].config(bg=color[1])

colors = [[255, 255, 255] for i in range(5)]  # initialize colors to white
color_labels = []
color_buttons = []

# Right Side Part
R = Frame(root, borderwidth=6)
R.pack(side=RIGHT, fill=Y)

for i in range(5):
    label = Label(R, text=f"Class {i+1}:")
    label.grid(row=7+i, column=0, padx=5, pady=5)

    button = Button(R, text="Choose Color", command=lambda index=i: on_color_button(index))
    button.grid(row=7+i, column=1, padx=5, pady=5)

    color_buttons.append(button)

def classified_image():
    test = raw_image(tif_path)
    test1 = test.copy()
    test1['Label'] = Classifier(kernel.get())[2]
    test1.loc[test1['Label'] == 1, ['Band_1', 'Band_2', 'Band_3']] = colors[0]  # water
    test1.loc[test1['Label'] == 2, ['Band_1', 'Band_2', 'Band_3']] = colors[1]  # veg
    test1.loc[test1['Label'] == 3, ['Band_1', 'Band_2', 'Band_3']] = colors[2]  # urban
    test1.loc[test1['Label'] == 4, ['Band_1', 'Band_2', 'Band_3']] = colors[3]  # silt
    test1.loc[test1['Label'] == 5, ['Band_1', 'Band_2', 'Band_3']] = colors[4] # agri
    return test1.drop(['Label'], axis=1).to_numpy()

def plotc(im_size=(10, 10), leg_size=(9, 1.5)):
    array = classified_image()
    img = array.reshape((1151, 1151, 3))
    clr = np.array([colors[0],colors[1],colors[2],colors[3],colors[4]])
    labels = ['Water', 'Vegetation', 'Urban', 'Silt', 'Agriculture']
    fig = plt.figure(figsize=(im_size[0]+leg_size[0], im_size[1]))
    gs = fig.add_gridspec(1, 2, width_ratios=[im_size[0], leg_size[0]])
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(img)
    ax1.axis('off')
    ax2 = fig.add_subplot(gs[1])
    im = ax2.imshow([clr], aspect='auto')
    ax2.set_xticks(np.arange(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_yticks([])
    im_ratio = leg_size[1] / leg_size[0]
    ax2.set_aspect(im_ratio)
    plt.show()

def plotr():
    with rasterio.open(tif_path) as src:
        array = src.read()
        rows = src.height
        cols = src.width
        bands = src.count
        array = np.moveaxis(array, 0, -1)
        array = np.reshape(array, [rows*cols, bands])
        array = array.astype(np.uint8)
        array = array.reshape((1151,1151,4))
        x = np.dstack((array[:,:,2],array[:,:,1],array[:,:,0]))
        plt.imshow(x)

def create_report():
    # Run the classifier to get predictions
    y_test, y_pred, y_pred_all = Classifier(kernel.get())

    # Calculate metrics and generate report
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Create a new window to display the report
    report_window = Toplevel(root)
    report_window.title('Classification Report')

    # Add a label for accuracy
    accuracy_label = Label(report_window, text=f'Accuracy: {accuracy:.2f}')
    accuracy_label.pack()

    rep = Label((report_window),text = f'{report}')
    rep.pack()

    # Add a confusion matrix plot
    plt.figure()
    sns.heatmap(confusion, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion matrix')
    canvas = FigureCanvasTkAgg(plt.gcf(), master=report_window)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Add a text box for the classification report
    #report_text = Text(report_window, height=10, width=50)
    #report_text.insert(END, report)
    #report_text.pack()

###############################################################################################################################


# VARIABLES
C = IntVar()
deg = IntVar()
gamma = IntVar()
kernel = StringVar()
options = ['linear', 'poly', 'rbf', 'sigmoid']
kernel.set(options[0])

# LABELS
l0 = Label(R, text='Input Image:')
l1 = Label(R, text='Training Data:')
l2 = Label(R, text='Select Kernel Type')
l3 = Label(R,text='Regularization Parameter')
l4 = Label(R,text='Degree')
l5 = Label(R,text='Gamma')

# ENTRY
M1 = OptionMenu(R, kernel, *options)
E1 = Entry(R)
E2 = Entry(R)
E3 = Entry(R)

# BUTTONS
b0 = Button(R, text='Select Tiff File', command=select_imagefile)
b1 = Button(R, text='Select CSV file', command=select_file)
b2 = Button(R, text='View Raw Image', command=plotr)
b3 = Button(R, text='View Classified Image',command=plotc)
b4 = Button(R, text='Report',command=create_report)


# GRIDS
l0.grid(row=0, column=0, padx=10, pady=10, sticky='W')
l1.grid(row=1, column=0, padx=10, pady=10, sticky='W')
l2.grid(row=2, column=0, padx=10, pady=10, sticky='W')
l3.grid(row=3,column=0,padx=10,pady=10,sticky='W')
l4.grid(row=4,column=0,padx=10,pady=10,sticky='W')
l5.grid(row=5,column=0,padx=10,pady=10,sticky='W')


b0.grid(row=0, column=1, padx=10, pady=10)
b1.grid(row=1, column=1, padx=10, pady=10)
b2.grid(row=13, column=0, padx=10, pady=10)
b3.grid(row=13, column=1, padx=10, pady=10)
b4.grid(row=14, column=1, padx=10, pady=10)

M1.grid(row=2, column=1, padx=10, pady=10)

E1.grid(row=3,column=1,padx=10,pady=10)
E2.grid(row=4,column=1,padx=10,pady=10)
E3.grid(row=5,column=1,padx=10,pady=10)

###############################################################################################################################

# Left Side Part
L = Frame(root, bg='grey', borderwidth=6)
L.pack(side=LEFT, fill=Y, anchor='w', expand=True)

txt = Label(L, text='Welcome', fg='black', border=5)
txt.pack(fill=Y)
root.mainloop()
