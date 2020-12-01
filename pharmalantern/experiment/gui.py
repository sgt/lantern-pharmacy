import tkinter as tk

from PIL import ImageTk, Image

from pharmalantern.image import load_grayscale_image


def run_gui(image_filename) -> None:
    # Set up GUI
    window = tk.Tk()  # Makes main window
    window.title("фонарь-аптека")
    window.config(background="#FFFFFF")

    # Graphics window
    # imageFrame = tk.Frame(window, width=600, height=500)
    # imageFrame.grid(row=0, column=0, padx=10, pady=2)
    # imageFrame.pack(fill="both", expand=True)

    cv_img = load_grayscale_image(image_filename)
    img = ImageTk.PhotoImage(Image.fromarray(cv_img))
    panelA = tk.Label(window, image=img)
    panelA.image = img
    panelA.pack(fill=tk.BOTH, expand=tk.YES)

    window.mainloop()
