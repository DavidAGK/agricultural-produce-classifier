"""
Tkinter GUI for Agricultural Produce Classification Tool
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import threading
from src.classify import ImageClassifier, calculate_batch_statistics


class ProduceClassifierGUI:
    """Main GUI application for produce classification"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Agricultural Produce Classifier")
        self.root.geometry("1000x700")
        self.root.resizable(False, False)
        
        # Initialize classifier
        self.classifier = ImageClassifier()
        
        # Variables
        self.produce_type = tk.StringVar(value="mango")
        self.current_image_path = None
        self.current_result = None
        
        # Setup GUI
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI components"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="Agricultural Produce Quality Classifier",
            font=('Arial', 20, 'bold')
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=20)
        
        # Produce type selection
        type_frame = ttk.LabelFrame(main_frame, text="Select Produce Type", padding="10")
        type_frame.grid(row=1, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        
        ttk.Radiobutton(
            type_frame,
            text="Mango",
            variable=self.produce_type,
            value="mango"
        ).grid(row=0, column=0, padx=20)
        
        ttk.Radiobutton(
            type_frame,
            text="Plantain",
            variable=self.produce_type,
            value="plantain"
        ).grid(row=0, column=1, padx=20)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=20)
        
        ttk.Button(
            button_frame,
            text="Select Image",
            command=self.select_image
        ).grid(row=0, column=0, padx=10)
        
        ttk.Button(
            button_frame,
            text="Select Folder",
            command=self.select_folder
        ).grid(row=0, column=1, padx=10)
        
        ttk.Button(
            button_frame,
            text="Classify",
            command=self.classify_current
        ).grid(row=0, column=2, padx=10)
        
        # Image display frame
        self.image_frame = ttk.LabelFrame(main_frame, text="Image Preview", padding="10")
        self.image_frame.grid(row=3, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.image_label = ttk.Label(self.image_frame, text="No image selected")
        self.image_label.grid(row=0, column=0)
        
        # Results frame
        self.results_frame = ttk.LabelFrame(main_frame, text="Classification Results", padding="10")
        self.results_frame.grid(row=3, column=2, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.results_text = tk.Text(self.results_frame, width=40, height=20)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar for results
        scrollbar = ttk.Scrollbar(self.results_frame, command=self.results_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.results_text.config(yscrollcommand=scrollbar.set)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def select_image(self):
        """Open file dialog to select an image"""
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
            
    def select_folder(self):
        """Open folder dialog to select a folder for batch processing"""
        folder_path = filedialog.askdirectory(title="Select a folder containing images")
        
        if folder_path:
            self.status_var.set("Processing batch...")
            # Run batch processing in a separate thread
            thread = threading.Thread(
                target=self.process_batch,
                args=(folder_path,)
            )
            thread.start()
            
    def display_image(self, image_path):
        """Display image in the GUI"""
        try:
            # Open and resize image
            image = Image.open(image_path)
            image.thumbnail((400, 400), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Update label
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {str(e)}")
            
    def classify_current(self):
        """Classify the currently selected image"""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please select an image first")
            return
            
        try:
            self.status_var.set("Classifying...")
            
            # Classify image
            result, processed_img = self.classifier.classify_image(
                self.current_image_path,
                self.produce_type.get()
            )
            
            # Display results
            self.display_results(result)
            
            # Show processed image with result
            result_img = self.classifier.visualize_result(processed_img, result)
            cv2.imshow("Classification Result", result_img)
            
            self.status_var.set("Classification complete")
            
        except Exception as e:
            messagebox.showerror("Error", f"Classification failed: {str(e)}")
            self.status_var.set("Error occurred")
            
    def display_results(self, result):
        """Display classification results in the text widget"""
        self.results_text.delete(1.0, tk.END)
        
        self.results_text.insert(tk.END, "=== Classification Result ===\n\n")
        self.results_text.insert(tk.END, f"Quality: {result['quality']}\n")
        self.results_text.insert(tk.END, f"Good Percentage: {result['percentage']:.2f}%\n")
        self.results_text.insert(tk.END, f"Confidence: {result['confidence']:.4f}\n\n")
        
        # Add quality description
        if result['quality'] == "Good":
            description = "This produce is of good quality and suitable for transportation and processing."
        elif result['quality'] == "Fair":
            description = "This produce is ripe and should be used within 24-48 hours to avoid spoilage."
        else:
            description = "This produce is of poor quality and not suitable for transportation."
            
        self.results_text.insert(tk.END, f"Description:\n{description}\n")
        
    def process_batch(self, folder_path):
        """Process all images in a folder"""
        try:
            # Get batch results
            results = self.classifier.classify_batch(folder_path, self.produce_type.get())
            
            # Calculate statistics
            stats = calculate_batch_statistics(results)
            
            # Display batch results
            self.display_batch_results(results, stats)
            
            self.status_var.set(f"Batch processing complete: {stats['total']} images processed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Batch processing failed: {str(e)}")
            self.status_var.set("Error occurred")
            
    def display_batch_results(self, results, stats):
        """Display batch processing results"""
        self.results_text.delete(1.0, tk.END)
        
        self.results_text.insert(tk.END, "=== Batch Processing Results ===\n\n")
        self.results_text.insert(tk.END, f"Total Images: {stats['total']}\n")
        self.results_text.insert(tk.END, f"Good: {stats['good']} ({stats['good_percentage']:.1f}%)\n")
        self.results_text.insert(tk.END, f"Fair: {stats['fair']} ({stats['fair_percentage']:.1f}%)\n")
        self.results_text.insert(tk.END, f"Bad: {stats['bad']} ({stats['bad_percentage']:.1f}%)\n")
        
        if stats['errors'] > 0:
            self.results_text.insert(tk.END, f"Errors: {stats['errors']}\n")
            
        self.results_text.insert(tk.END, "\n=== Individual Results ===\n\n")
        
        for result in results:
            self.results_text.insert(
                tk.END,
                f"{result['filename']}: {result['quality']} ({result['percentage']:.1f}%)\n"
            )
            
    def run(self):
        """Run the GUI application"""
        self.root.mainloop()


if __name__ == "__main__":
    app = ProduceClassifierGUI()
    app.run()
