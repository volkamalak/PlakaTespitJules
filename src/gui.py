import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from src.detector import LicensePlateDetector
import os

class LicensePlateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("License Plate Detector")
        self.root.geometry("1000x700")

        # Initialize detector
        try:
            self.detector = LicensePlateDetector()
        except Exception as e:
            print(f"Failed to initialize detector: {e}")

        self.current_image_path = None
        self.original_pil_image = None
        self.processed_pil_image = None

        # Setup UI
        self.create_widgets()

    def create_widgets(self):
        # Top Control Frame
        control_frame = tk.Frame(self.root, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        btn_load = tk.Button(control_frame, text="Load Image", command=self.load_image, width=15)
        btn_load.pack(side=tk.LEFT, padx=20)

        btn_detect = tk.Button(control_frame, text="Run Program", command=self.run_detection, width=15, bg="#dddddd")
        btn_detect.pack(side=tk.LEFT, padx=20)

        # Main Content Frame (Split View)
        content_frame = tk.Frame(self.root)
        content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left Panel (Original)
        self.left_panel = tk.LabelFrame(content_frame, text="Original Image")
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.lbl_original = tk.Label(self.left_panel, text="No Image Loaded")
        self.lbl_original.pack(fill=tk.BOTH, expand=True)

        # Right Panel (Processed)
        self.right_panel = tk.LabelFrame(content_frame, text="Processed Image")
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        self.lbl_processed = tk.Label(self.right_panel, text="No Result Yet")
        self.lbl_processed.pack(fill=tk.BOTH, expand=True)

        # Bottom Info Frame
        info_frame = tk.Frame(self.root, height=150, pady=10)
        info_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10)

        # Info Labels
        tk.Label(info_frame, text="Detection Info:", font=("Arial", 10, "bold")).pack(anchor=tk.W)

        self.lbl_time = tk.Label(info_frame, text="Processing Time: N/A")
        self.lbl_time.pack(anchor=tk.W)

        self.txt_coords = tk.Text(info_frame, height=5)
        self.txt_coords.pack(fill=tk.X, pady=5)
        self.txt_coords.insert(tk.END, "Coordinates will appear here...")
        self.txt_coords.config(state=tk.DISABLED)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if not file_path:
            return

        self.current_image_path = file_path

        try:
            image = Image.open(file_path)
            self.original_pil_image = image

            # Resize for display
            display_image = self.resize_image_for_display(image)
            tk_image = ImageTk.PhotoImage(display_image)

            self.lbl_original.config(image=tk_image, text="")
            self.lbl_original.image = tk_image # Keep reference

            # Clear previous results
            self.lbl_processed.config(image="", text="No Result Yet")
            self.txt_coords.config(state=tk.NORMAL)
            self.txt_coords.delete(1.0, tk.END)
            self.txt_coords.insert(tk.END, "Ready to detect.")
            self.txt_coords.config(state=tk.DISABLED)
            self.lbl_time.config(text="Processing Time: N/A")

        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {e}")

    def run_detection(self):
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        self.lbl_processed.config(text="Processing...")
        self.root.update()

        try:
            processed_img, coords, time_taken = self.detector.detect(self.current_image_path)

            # Store processed image
            self.processed_pil_image = processed_img

            # Display processed image
            display_image = self.resize_image_for_display(processed_img)
            tk_image = ImageTk.PhotoImage(display_image)

            self.lbl_processed.config(image=tk_image, text="")
            self.lbl_processed.image = tk_image

            # Update Info
            self.lbl_time.config(text=f"Processing Time: {time_taken:.4f} seconds")

            self.txt_coords.config(state=tk.NORMAL)
            self.txt_coords.delete(1.0, tk.END)

            if not coords:
                self.txt_coords.insert(tk.END, "No plates detected.")
            else:
                for item in coords:
                    line = f"Label: {item['label']}, Conf: {item['confidence']:.2f}, Box: {item['bbox']}\n"
                    self.txt_coords.insert(tk.END, line)

            self.txt_coords.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {e}")
            self.lbl_processed.config(text="Error")

    def resize_image_for_display(self, image):
        # Resize image to fit in 450x450 box (approx) while keeping aspect ratio
        max_size = (450, 450)

        # Create a copy to not modify original
        img_copy = image.copy()
        img_copy.thumbnail(max_size, Image.LANCZOS)
        return img_copy

if __name__ == "__main__":
    root = tk.Tk()
    app = LicensePlateApp(root)
    root.mainloop()
