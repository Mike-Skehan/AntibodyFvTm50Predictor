import pipelines
import customtkinter as ctk


def predict_tm50(heavy, light):

    seq = [heavy,light]
    return pipelines.svm_pipe.predict(seq)


# GUI using tkinter
root = ctk.CTk()
root.title("AntiBERTy")
root.geometry("500x500")

# Create a label
heavy_label = ctk.CTkLabel(root, text="Enter heavy chain sequence: ")
heavy_label.pack()

# Create an entry box
heavy_entry = ctk.CTkEntry(root)
heavy_entry.pack()

# Create a label
light_label = ctk.CTkLabel(root, text="Enter light chain sequence: ")
light_label.pack()

# Create an entry box
light_entry = ctk.CTkEntry(root)
light_entry.pack()

result = ctk.CTkLabel(root, text="")
result.pack()


def get_result():
    tm50 = (predict_tm50(heavy_entry.get(), light_entry.get()))
    rounded_tm50 = round(*tm50, 2)
    output_text = "The predicted tm50 value is: " + str(rounded_tm50) + "°C"
    result.configure(text = output_text)


# Create a submit button which will use the predict_tm50 function
submit_button = ctk.CTkButton(root, text="Submit", command=get_result)
submit_button.pack()

# Run the mainloop
root.mainloop()


