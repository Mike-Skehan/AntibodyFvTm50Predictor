import pipelines
import tkinter as tk


def predict_tm50(heavy, light):

    seq = [heavy,light]
    return pipelines.svm_pipe.predict(seq)


#heavy = input("Enter heavy chain sequence: ")
#light = input("Enter light chain sequence: ")

#result = predict_tm50(heavy, light)

#print("The predicted tm50 value is: ", round(*result, 2))

# GUI using tkinter
root = tk.Tk()
root.title("AntiBERTy")
root.geometry("500x500")

# Create a label
heavy_label = tk.Label(root, text="Enter heavy chain sequence: ")
heavy_label.pack()

# Create an entry box
heavy_entry = tk.Entry(root)
heavy_entry.pack()

# Create a label
light_label = tk.Label(root, text="Enter light chain sequence: ")
light_label.pack()

# Create an entry box
light_entry = tk.Entry(root)
light_entry.pack()

result = tk.Label(root, text="")
result.pack()


def get_result():
    tm50 = (predict_tm50(heavy_entry.get(), light_entry.get()))
    rounded_tm50 = round(*tm50, 2)
    output_text = "The predicted tm50 value is: " + str(rounded_tm50) + "°C"
    result.config(text = output_text)


# Create a submit button which will use the predict_tm50 function
submit_button = tk.Button(root, text="Submit", command=get_result)
submit_button.pack()

# Run the mainloop
root.mainloop()
