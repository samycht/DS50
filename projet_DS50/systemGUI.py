import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import joblib
import numpy as np
mapping = {
    0:'adenocarcinoma      ',
    1:'large.cell.carcinoma',
    2:'normal! congratulations',
    3:'squamous.cell.carcinoma'
}
#this function is used to get your picture
def open_image():
    imagepath = filedialog.askopenfilename(
        initialdir="/",
        title="Select file",
        filetypes=(("Image files", ("*.jpg", "*.jpeg", "*.png")), ("All files", "*.*"))
    )
    if imagepath:
        image = Image.open(imagepath)
        image_test = image
        image.thumbnail((300, 200))
        image = ImageTk.PhotoImage(image)
        image_label.configure(image=image)
        image_label.image = image
        new_label = predict_image(image_test)
        Label_root = tk.Label(window,text='the result of your CT is: '+new_label,justify=tk.LEFT,compound = tk.CENTER,font=("Comic Sans MS",14),fg = "black")
        Label_root.configure(bg='#1E90FF')
        Label_root.place(x=90,y=350)
#this function is used to get the result of predict
def predict_image(image):
    model = joblib.load('DT1.dat')
    width, height = 150, 150
    # prepare your picture
    image = image.convert('RGB')
    image = image.resize((width, height))
    image = np.array(image) / 255.0
    # transform the picture into right size
    image_reshaped = image.reshape((1,150, 150, 3))
    # do the prediction
    predictions = model.predict(image_reshaped)
    # deal with the result
    predicted_class = np.argmax(predictions[0])
    return mapping[predicted_class]

def set_window_style():
    # set your title here
    window.title("pneumonia-predict-system")
    # set the parameters of window
    window.geometry("600x400")
    window.configure(bg='LightSteelBlue')
    Label_root = tk.Label(window,text="this is a pneumonia predict system\n   please enter the picture of CT",justify=tk.LEFT,compound = tk.CENTER,font=("Comic Sans MS",16),fg = "black")
    Label_root.configure(bg='LightSteelBlue')
    Label_root.place(x=140,y=280)

# create window
window = tk.Tk()
# create the button to choose picture
button = tk.Button(window, text="click to choose CT",bg="#1E90FF", command=open_image)
button.pack(pady=10)
#
set_window_style()
#initial your image label
image_label = tk.Label(window)
image_label.pack()
image = Image.open("AI.jpg")
image.thumbnail((300, 230))
image = ImageTk.PhotoImage(image)
image_label.configure(image=image)
# run the window
window.mainloop()
