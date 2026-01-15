import tkinter as tk
from src.gui import LicensePlateApp

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = LicensePlateApp(root)
        root.mainloop()
    except Exception as e:
        print(f"An error occurred: {e}")
        # Keep terminal open if running as script/exe
        input("Press Enter to exit...")
