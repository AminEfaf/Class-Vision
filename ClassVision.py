import tkinter as tk
from tkinter import ttk, messagebox
import os
import cv2
import csv
import numpy as np
from PIL import ImageTk, Image
import pandas as pd
import datetime
import time
import pyttsx3
from glob import glob
import shutil
import threading

class AttendanceSystem:
    def __init__(self):
        # Configuration
        self.haar_path = "haarcascade_frontalface_default.xml"
        self.trainer_path = "./TrainingImageLabel/Trainner.yml"
        self.train_path = "./TrainingImage"
        self.student_details_path = "./StudentDetails/studentdetails.csv"
        self.attendance_path = "./Attendance"
        
        # Attendance control variables
        self.attendance_running = False
        self.attendance_thread = None
        
        # Create directories if they don't exist
        for path in [self.train_path, "./StudentDetails", self.attendance_path, "./TrainingImageLabel"]:
            os.makedirs(path, exist_ok=True)
        
        # Initialize text-to-speech
        self.tts_engine = pyttsx3.init()
        
        # Setup main window
        self.setup_main_window()
    
    def speak(self, text):
        """Text to speech function"""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
    
    def setup_main_window(self):
        """Setup the main application window"""
        self.root = tk.Tk()
        self.root.title("CLASS VISION - Face Recognition Attendance System")
        self.root.geometry("900x600")
        self.root.configure(bg="#1c1c1c")
        
        # Title
        title_frame = tk.Frame(self.root, bg="#1c1c1c")
        title_frame.pack(fill=tk.X, pady=20)
        
        tk.Label(title_frame, text="CLASS VISION", 
                bg="#1c1c1c", fg="yellow", 
                font=("Verdana", 30, "bold")).pack()
        
        tk.Label(title_frame, text="Face Recognition Attendance System", 
                bg="#1c1c1c", fg="white", 
                font=("Verdana", 16)).pack()
        
        # Buttons frame
        buttons_frame = tk.Frame(self.root, bg="#1c1c1c")
        buttons_frame.pack(expand=True)
        
        # Create buttons
        button_style = {"font": ("Verdana", 14), "bg": "#333333", "fg": "yellow",
                       "width": 20, "height": 2, "relief": tk.RIDGE, "bd": 3}
        
        tk.Button(buttons_frame, text="Add Student", 
                 command=self.register_student_ui, **button_style).pack(pady=10)
        
        tk.Button(buttons_frame, text="Check Roll", 
                 command=self.take_attendance_ui, **button_style).pack(pady=10)
        
        tk.Button(buttons_frame, text="Check Records", 
                 command=self.view_attendance_ui, **button_style).pack(pady=10)
        
        tk.Button(buttons_frame, text="Delete Student", 
                 command=self.delete_student_ui, **button_style).pack(pady=10)
        
        tk.Button(buttons_frame, text="EXIT", 
                 command=self.root.quit, **button_style).pack(pady=20)
    
    def delete_student_ui(self):
        """UI for deleting students"""
        delete_window = tk.Toplevel(self.root)
        delete_window.title("Delete Student")
        delete_window.geometry("600x500")
        delete_window.configure(bg="#1c1c1c")
        delete_window.grab_set()
        
        # Title
        tk.Label(delete_window, text="Delete Student", 
                bg="#1c1c1c", fg="red", 
                font=("Verdana", 24, "bold")).pack(pady=20)
        
        # Load existing students
        students_df = self.load_student_details()
        
        if students_df.empty:
            tk.Label(delete_window, text="No students found to delete!", 
                    bg="#1c1c1c", fg="yellow", 
                    font=("Verdana", 16)).pack(pady=50)
            return
        
        # Student selection frame
        selection_frame = tk.Frame(delete_window, bg="#1c1c1c")
        selection_frame.pack(pady=20, fill=tk.BOTH, expand=True)
        
        tk.Label(selection_frame, text="Select Student to Delete:", 
                bg="#1c1c1c", fg="yellow", 
                font=("Verdana", 14)).pack()
        
        # Create listbox with scrollbar
        listbox_frame = tk.Frame(selection_frame, bg="#1c1c1c")
        listbox_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Listbox
        student_listbox = tk.Listbox(listbox_frame, 
                                   font=("Verdana", 12), 
                                   bg="#333333", fg="yellow",
                                   selectmode=tk.SINGLE,
                                   height=10)
        
        # Scrollbar
        scrollbar = tk.Scrollbar(listbox_frame, orient=tk.VERTICAL)
        student_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=student_listbox.yview)
        
        # Pack listbox and scrollbar
        student_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate listbox with students
        for _, student in students_df.iterrows():
            display_text = f"{student['Enrollment']} - {student['Name']}"
            student_listbox.insert(tk.END, display_text)
        
        # Status label
        status_label = tk.Label(delete_window, text="", 
                               bg="#1c1c1c", fg="yellow", 
                               font=("Verdana", 12))
        status_label.pack(pady=10)
        
        # Buttons frame
        button_frame = tk.Frame(delete_window, bg="#1c1c1c")
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="Delete Selected", 
                 command=lambda: self.delete_selected_student(student_listbox, students_df, status_label),
                 font=("Verdana", 12), bg="#cc0000", fg="white", 
                 width=15, height=2).pack(side=tk.LEFT, padx=10)
        
        tk.Button(button_frame, text="Delete ALL Students", 
                 command=lambda: self.delete_all_students(status_label),
                 font=("Verdana", 12), bg="#990000", fg="white", 
                 width=18, height=2).pack(side=tk.LEFT, padx=10)
        
        tk.Button(button_frame, text="Cancel", 
                 command=delete_window.destroy,
                 font=("Verdana", 12), bg="#333333", fg="yellow", 
                 width=15, height=2).pack(side=tk.LEFT, padx=10)
    
    def load_student_details(self):
        """Load student details from CSV file"""
        try:
            if os.path.exists(self.student_details_path):
                df = pd.read_csv(self.student_details_path)
                # Handle different column name variations
                if 'enrollment' in df.columns:
                    df.rename(columns={'enrollment': 'Enrollment'}, inplace=True)
                if 'name' in df.columns:
                    df.rename(columns={'name': 'Name'}, inplace=True)
                return df
            else:
                return pd.DataFrame(columns=['Enrollment', 'Name'])
        except Exception as e:
            print(f"Error loading student details: {e}")
            return pd.DataFrame(columns=['Enrollment', 'Name'])
    
    def delete_selected_student(self, listbox, students_df, status_label):
        """Delete the selected student"""
        selection = listbox.curselection()
        
        if not selection:
            msg = "Please select a student to delete"
            status_label.config(text=msg)
            self.speak(msg)
            return
        
        # Get selected student
        selected_index = selection[0]
        selected_student = students_df.iloc[selected_index]
        enrollment = str(selected_student['Enrollment'])
        name = selected_student['Name']
        
        # Confirmation dialog
        if not messagebox.askyesno("Confirm Deletion", 
                                  f"Are you sure you want to delete:\n{enrollment} - {name}?"):
            return
        
        try:
            # Delete from CSV file
            students_df = students_df.drop(students_df.index[selected_index])
            students_df.to_csv(self.student_details_path, index=False)
            
            # Delete training images folder
            self.delete_student_images(enrollment, name)
            
            # Remove from listbox
            listbox.delete(selected_index)
            
            msg = f"Successfully deleted {name} ({enrollment})"
            status_label.config(text=msg)
            self.speak(msg)
            
            # If model exists, suggest retraining
            if os.path.exists(self.trainer_path):
                if messagebox.askyesno("Retrain Model", 
                                     "Student deleted. Do you want to retrain the model now?"):
                    self.train_model(status_label)
            
        except Exception as e:
            msg = f"Error deleting student: {str(e)}"
            status_label.config(text=msg)
            self.speak(msg)
    
    def delete_all_students(self, status_label):
        """Delete all students"""
        # Confirmation dialog
        if not messagebox.askyesno("Confirm Deletion", 
                                  "Are you sure you want to delete ALL students?\nThis action cannot be undone!"):
            return
        
        try:
            # Delete CSV file
            if os.path.exists(self.student_details_path):
                os.remove(self.student_details_path)
            
            # Delete all training images
            if os.path.exists(self.train_path):
                shutil.rmtree(self.train_path)
                os.makedirs(self.train_path, exist_ok=True)
            
            # Delete trained model
            if os.path.exists(self.trainer_path):
                os.remove(self.trainer_path)
            
            msg = "All students deleted successfully"
            status_label.config(text=msg)
            self.speak(msg)
            
        except Exception as e:
            msg = f"Error deleting all students: {str(e)}"
            status_label.config(text=msg)
            self.speak(msg)
    
    def delete_student_images(self, enrollment, name):
        """Delete training images for a specific student"""
        try:
            # Look for student folder with different naming patterns
            possible_names = [
                f"{enrollment}_{name}",
                f"{name}_{enrollment}",
                enrollment,
                name
            ]
            
            for folder_name in os.listdir(self.train_path):
                folder_path = os.path.join(self.train_path, folder_name)
                if os.path.isdir(folder_path):
                    # Check if folder matches any possible naming pattern
                    if any(possible_name.lower() in folder_name.lower() for possible_name in possible_names):
                        shutil.rmtree(folder_path)
                        print(f"Deleted folder: {folder_path}")
                        break
            
            # Also delete individual image files that might match
            for file in os.listdir(self.train_path):
                if os.path.isfile(os.path.join(self.train_path, file)):
                    if enrollment in file or name.lower() in file.lower():
                        os.remove(os.path.join(self.train_path, file))
                        print(f"Deleted file: {file}")
                        
        except Exception as e:
            print(f"Error deleting student images: {e}")
    
    def register_student_ui(self):
        """UI for student registration"""
        reg_window = tk.Toplevel(self.root)
        reg_window.title("Register Student")
        reg_window.geometry("600x500")
        reg_window.configure(bg="#1c1c1c")
        reg_window.grab_set()
        
        # Title
        tk.Label(reg_window, text="Register Your Face", 
                bg="#1c1c1c", fg="green", 
                font=("Verdana", 24, "bold")).pack(pady=20)
        
        # Input frame
        input_frame = tk.Frame(reg_window, bg="#1c1c1c")
        input_frame.pack(pady=20)
        
        # Enrollment number
        tk.Label(input_frame, text="Enrollment No:", 
                bg="#1c1c1c", fg="yellow", 
                font=("Verdana", 14)).grid(row=0, column=0, padx=10, pady=10, sticky="e")
        
        enrollment_var = tk.StringVar()
        enrollment_entry = tk.Entry(input_frame, textvariable=enrollment_var,
                                   font=("Verdana", 14), width=20, 
                                   bg="#333333", fg="yellow")
        enrollment_entry.grid(row=0, column=1, padx=10, pady=10)
        
        # Name
        tk.Label(input_frame, text="Name:", 
                bg="#1c1c1c", fg="yellow", 
                font=("Verdana", 14)).grid(row=1, column=0, padx=10, pady=10, sticky="e")
        
        name_var = tk.StringVar()
        name_entry = tk.Entry(input_frame, textvariable=name_var,
                             font=("Verdana", 14), width=20, 
                             bg="#333333", fg="yellow")
        name_entry.grid(row=1, column=1, padx=10, pady=10)
        
        # Status label
        status_label = tk.Label(reg_window, text="", 
                               bg="#1c1c1c", fg="yellow", 
                               font=("Verdana", 12))
        status_label.pack(pady=20)
        
        # Buttons
        button_frame = tk.Frame(reg_window, bg="#1c1c1c")
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="Capture Images", 
                 command=lambda: self.capture_images(enrollment_var.get(), name_var.get(), status_label),
                 font=("Verdana", 12), bg="#333333", fg="yellow", 
                 width=15, height=2).pack(side=tk.LEFT, padx=10)
        
        tk.Button(button_frame, text="Train Model", 
                 command=lambda: self.train_model(status_label),
                 font=("Verdana", 12), bg="#333333", fg="yellow", 
                 width=15, height=2).pack(side=tk.LEFT, padx=10)
    
    def capture_images(self, enrollment, name, status_label):
        """Capture student images for training"""
        if not enrollment or not name:
            msg = "Please enter both enrollment number and name"
            status_label.config(text=msg)
            self.speak(msg)
            return
        
        try:
            # Create directory for student
            student_dir = f"{enrollment}_{name}"
            path = os.path.join(self.train_path, student_dir)
            
            if os.path.exists(path):
                msg = "Student already exists"
                status_label.config(text=msg)
                self.speak(msg)
                return
                
            os.makedirs(path)
            
            # Start capturing
            cam = cv2.VideoCapture(0)
            detector = cv2.CascadeClassifier(self.haar_path)
            sample_count = 0
            
            status_label.config(text="Capturing images... Press 'q' to stop")
            
            while True:
                ret, img = cam.read()
                if not ret:
                    break
                    
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    sample_count += 1
                    
                    # Save face image
                    face_img = gray[y:y + h, x:x + w]
                    filename = f"{path}/{name}_{enrollment}_{sample_count}.jpg"
                    cv2.imwrite(filename, face_img)
                    
                    cv2.putText(img, f"Count: {sample_count}", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
                cv2.imshow("Capturing Images", img)
                
                if cv2.waitKey(1) & 0xFF == ord('q') or sample_count >= 100:
                    break
            
            cam.release()
            cv2.destroyAllWindows()
            
            # Save student details
            # Check if file exists and has headers
            file_exists = os.path.exists(self.student_details_path)
            
            with open(self.student_details_path, 'a', newline='') as file:
                writer = csv.writer(file)
                
                # If file doesn't exist or is empty, write headers first
                if not file_exists or os.path.getsize(self.student_details_path) == 0:
                    writer.writerow(['Enrollment', 'Name'])
                
                writer.writerow([enrollment, name])
            
            msg = f"Successfully captured {sample_count} images for {name}"
            status_label.config(text=msg)
            self.speak(msg)
            
        except Exception as e:
            msg = f"Error: {str(e)}"
            status_label.config(text=msg)
            self.speak(msg)
    
    def train_model(self, status_label):
        """Train the face recognition model"""
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            faces, ids = self.get_images_and_labels()
            
            if len(faces) == 0:
                msg = "No training images found"
                status_label.config(text=msg)
                self.speak(msg)
                return
            
            recognizer.train(faces, np.array(ids))
            recognizer.save(self.trainer_path)
            
            msg = "Model trained successfully"
            status_label.config(text=msg)
            self.speak(msg)
            
        except Exception as e:
            msg = f"Training error: {str(e)}"
            status_label.config(text=msg)
            self.speak(msg)
    
    def get_images_and_labels(self):
        """Get training images and corresponding labels"""
        faces = []
        ids = []
        
        if not os.path.exists(self.train_path):
            return faces, ids
        
        for student_dir in os.listdir(self.train_path):
            student_path = os.path.join(self.train_path, student_dir)
            if not os.path.isdir(student_path):
                continue
                
            for image_file in os.listdir(student_path):
                image_path = os.path.join(student_path, image_file)
                
                try:
                    # Extract ID from filename
                    parts = image_file.split('_')
                    if len(parts) >= 2:
                        student_id = int(parts[1])
                        
                        # Load and process image
                        pil_image = Image.open(image_path).convert('L')
                        image_np = np.array(pil_image, 'uint8')
                        
                        faces.append(image_np)
                        ids.append(student_id)
                except (ValueError, IndexError):
                    continue
        
        return faces, ids
    
    def take_attendance_ui(self):
        """UI for taking attendance with stop button"""
        att_window = tk.Toplevel(self.root)
        att_window.title("Take Attendance")
        att_window.geometry("500x400")
        att_window.configure(bg="#1c1c1c")
        att_window.grab_set()
        
        tk.Label(att_window, text="Take Attendance", 
                bg="#1c1c1c", fg="green", 
                font=("Verdana", 20, "bold")).pack(pady=20)
        
        # Subject entry
        input_frame = tk.Frame(att_window, bg="#1c1c1c")
        input_frame.pack(pady=20)
        
        tk.Label(input_frame, text="Subject Name:", 
                bg="#1c1c1c", fg="yellow", 
                font=("Verdana", 14)).pack()
        
        subject_var = tk.StringVar()
        subject_entry = tk.Entry(input_frame, textvariable=subject_var,
                                font=("Verdana", 16), width=20, 
                                bg="#333333", fg="yellow")
        subject_entry.pack(pady=10)
        
        # Status label
        status_label = tk.Label(att_window, text="", 
                               bg="#1c1c1c", fg="yellow", 
                               font=("Verdana", 12))
        status_label.pack(pady=20)
        
        # Buttons frame
        button_frame = tk.Frame(att_window, bg="#1c1c1c")
        button_frame.pack(pady=20)
        
        # Start attendance button
        start_button = tk.Button(button_frame, text="Start Attendance", 
                                command=lambda: self.start_attendance(subject_var.get(), status_label, att_window, start_button, stop_button),
                                font=("Verdana", 14), bg="#00cc00", fg="white", 
                                width=15, height=2)
        start_button.pack(side=tk.LEFT, padx=10)
        
        # Stop attendance button
        stop_button = tk.Button(button_frame, text="Stop Attendance", 
                               command=lambda: self.stop_attendance(status_label, start_button, stop_button),
                               font=("Verdana", 14), bg="#cc0000", fg="white", 
                               width=15, height=2, state=tk.DISABLED)
        stop_button.pack(side=tk.LEFT, padx=10)
            
    def start_attendance(self, subject, status_label, parent_window, start_button, stop_button):
        """Start the attendance process in a separate thread"""
        if not subject:
            msg = "Please enter subject name"
            status_label.config(text=msg)
            self.speak(msg)
            return
        
        if self.attendance_running:
            msg = "Attendance already running"
            status_label.config(text=msg)
            return
        
        # Update button states
        start_button.config(state=tk.DISABLED)
        stop_button.config(state=tk.NORMAL)
        
        # Set running flag
        self.attendance_running = True
        
        # Start attendance in a separate thread
        self.attendance_thread = threading.Thread(
            target=self._attendance_worker,
            args=(subject, status_label, parent_window, start_button, stop_button)
        )
        self.attendance_thread.daemon = True
        self.attendance_thread.start()
    
    def stop_attendance(self, status_label, start_button, stop_button):
        """Stop the attendance process"""
        self.attendance_running = False
        
        # Update button states
        start_button.config(state=tk.NORMAL)
        stop_button.config(state=tk.DISABLED)
        
        status_label.config(text="Stopping attendance...")
        
        # Wait for thread to finish
        if self.attendance_thread and self.attendance_thread.is_alive():
            self.attendance_thread.join(timeout=2)
    
    def _attendance_worker(self, subject, status_label, parent_window, start_button, stop_button):
        """Worker function for attendance process"""
        try:
            # Load trained model
            if not os.path.exists(self.trainer_path):
                self.root.after(0, lambda: status_label.config(text="Model not found. Please train the model first."))
                self.root.after(0, lambda: self.speak("Model not found. Please train the model first."))
                return
            
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(self.trainer_path)
            face_cascade = cv2.CascadeClassifier(self.haar_path)
            
            # Load student details
            if not os.path.exists(self.student_details_path):
                self.root.after(0, lambda: status_label.config(text="No student details found"))
                self.root.after(0, lambda: self.speak("No student details found"))
                return
            
            try:
                df = pd.read_csv(self.student_details_path)
                
                if df.empty:
                    self.root.after(0, lambda: status_label.config(text="Student details file is empty. Please register students first."))
                    self.root.after(0, lambda: self.speak("Student details file is empty. Please register students first."))
                    return
                
                # Ensure proper column names
                if 'Enrollment' not in df.columns and 'enrollment' in df.columns:
                    df.rename(columns={'enrollment': 'Enrollment'}, inplace=True)
                if 'Name' not in df.columns and 'name' in df.columns:
                    df.rename(columns={'name': 'Name'}, inplace=True)
                
                if 'Enrollment' not in df.columns or 'Name' not in df.columns:
                    self.root.after(0, lambda: status_label.config(text="Invalid student details format. Required columns: Enrollment, Name"))
                    self.root.after(0, lambda: self.speak("Invalid student details format"))
                    return
                
                df['Enrollment'] = pd.to_numeric(df['Enrollment'], errors='coerce')
                df = df.dropna(subset=['Enrollment'])
                
                print(f"Loaded {len(df)} students from database")
                
            except Exception as csv_error:
                self.root.after(0, lambda: status_label.config(text=f"Error reading student details: {str(csv_error)}"))
                self.root.after(0, lambda: self.speak("Error reading student details"))
                return
            
            # Initialize camera
            cam = cv2.VideoCapture(0)
            
            if not cam.isOpened():
                self.root.after(0, lambda: status_label.config(text="Cannot access camera"))
                self.root.after(0, lambda: self.speak("Cannot access camera"))
                return
            
            # Initialize attendance tracking
            attendance_list = []
            self.root.after(0, lambda: status_label.config(text="Taking attendance... Click 'Stop' when done"))
            
            while self.attendance_running:
                ret, frame = cam.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.2, 5)
                
                for (x, y, w, h) in faces:
                    face_roi = gray[y:y + h, x:x + w]
                    student_id, confidence = recognizer.predict(face_roi)
                    
                    if confidence < 60:
                        try:
                            student_matches = df[df['Enrollment'] == student_id]
                            
                            if not student_matches.empty:
                                name = student_matches['Name'].iloc[0]
                                label = f"{student_id}-{name}"
                                
                                # Add to attendance if not already present
                                if student_id not in [att[0] for att in attendance_list]:
                                    attendance_list.append([student_id, name])
                                    print(f"Added to attendance: {student_id} - {name}")
                                    # Update status on main thread
                                    self.root.after(0, lambda n=name: status_label.config(text=f"Recognized: {n}"))
                                
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                cv2.putText(frame, label, (x, y-10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            else:
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                                cv2.putText(frame, f"ID:{student_id} Not Found", (x, y-10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        except Exception as lookup_error:
                            print(f"Error looking up student: {lookup_error}")
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            cv2.putText(frame, "Lookup Error", (x, y-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, f"Unknown ({confidence:.1f})", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                cv2.imshow("Taking Attendance - Press 'q' to stop or use Stop button", frame)
                
                # Check for 'q' key press or stop flag
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.attendance_running = False
                    break
            
            # Cleanup
            cam.release()
            cv2.destroyAllWindows()
            
            # Save attendance if any students were recognized
            if attendance_list:
                self.save_attendance(subject, attendance_list)
                self.root.after(0, lambda: status_label.config(text=f"Attendance saved for {len(attendance_list)} students"))
                self.root.after(0, lambda: self.speak(f"Attendance saved for {len(attendance_list)} students"))
                
                # Show attendance table
                self.root.after(0, lambda: self.show_attendance_table(subject, attendance_list, parent_window))
            else:
                self.root.after(0, lambda: status_label.config(text="No students recognized"))
                self.root.after(0, lambda: self.speak("No students recognized"))
            
        except Exception as e:
            self.root.after(0, lambda: status_label.config(text=f"Error: {str(e)}"))
            self.root.after(0, lambda: self.speak(f"Error occurred during attendance"))
            print(f"Full error details: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Reset flags and button states
            self.attendance_running = False
            self.root.after(0, lambda: start_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: stop_button.config(state=tk.DISABLED))
    
    def save_attendance(self, subject, attendance_list):
        """Save attendance to CSV file"""
        # Create subject directory
        subject_dir = os.path.join(self.attendance_path, subject)
        os.makedirs(subject_dir, exist_ok=True)
        
        # Create filename with timestamp
        now = datetime.datetime.now()
        filename = f"{subject}_{now.strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        filepath = os.path.join(subject_dir, filename)
        
        # Create DataFrame and save
        df = pd.DataFrame(attendance_list, columns=['Enrollment', 'Name'])
        df[now.strftime('%Y-%m-%d')] = 1  # Mark as present
        df.to_csv(filepath, index=False)
    
    def show_attendance_table(self, subject, attendance_list, parent):
        """Display attendance in a table"""
        table_window = tk.Toplevel(parent)
        table_window.title(f"Attendance - {subject}")
        table_window.geometry("400x300")
        table_window.configure(bg="#1c1c1c")
        
        # Create treeview
        tree = ttk.Treeview(table_window, columns=('Enrollment', 'Name'), show='headings')
        tree.heading('Enrollment', text='Enrollment')
        tree.heading('Name', text='Name')
        
        for student in attendance_list:
            tree.insert('', tk.END, values=student)
        
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def view_attendance_ui(self):
        """UI for viewing attendance records"""
        view_window = tk.Toplevel(self.root)
        view_window.title("Check Records")
        view_window.geometry("500x300")
        view_window.configure(bg="#1c1c1c")
        view_window.grab_set()
        
        tk.Label(view_window, text="View Attendance Records", 
                bg="#1c1c1c", fg="green", 
                font=("Verdana", 20, "bold")).pack(pady=20)
        
        # Subject entry
        input_frame = tk.Frame(view_window, bg="#1c1c1c")
        input_frame.pack(pady=20)
        
        tk.Label(input_frame, text="Subject Name:", 
                bg="#1c1c1c", fg="yellow", 
                font=("Verdana", 14)).pack()
        
        subject_var = tk.StringVar()
        subject_entry = tk.Entry(input_frame, textvariable=subject_var,
                                font=("Verdana", 16), width=20, 
                                bg="#333333", fg="yellow")
        subject_entry.pack(pady=10)
        
        button_frame = tk.Frame(view_window, bg="#1c1c1c")
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="View Summary", 
                 command=lambda: self.view_attendance_summary(subject_var.get()),
                 font=("Verdana", 12), bg="#333333", fg="yellow", 
                 width=15, height=2).pack(side=tk.LEFT, padx=10)
        
        tk.Button(button_frame, text="Open Folder", 
                 command=lambda: self.open_attendance_folder(subject_var.get()),
                 font=("Verdana", 12), bg="#333333", fg="yellow", 
                 width=15, height=2).pack(side=tk.LEFT, padx=10)
    
    def view_attendance_summary(self, subject):
        """View attendance summary for a subject"""
        if not subject:
            self.speak("Please enter subject name")
            return
        
        try:
            subject_path = os.path.join(self.attendance_path, subject)
            if not os.path.exists(subject_path):
                self.speak("No attendance records found for this subject")
                return
            
            # Get all CSV files for the subject
            csv_files = glob(os.path.join(subject_path, f"{subject}_*.csv"))
            
            if not csv_files:
                self.speak("No attendance files found")
                return
            
            # Merge all attendance files
            dfs = [pd.read_csv(file) for file in csv_files]
            merged_df = dfs[0]
            
            for df in dfs[1:]:
                merged_df = merged_df.merge(df, on=['Enrollment', 'Name'], how='outer')
            
            merged_df.fillna(0, inplace=True)
            
            # Calculate attendance percentage
            date_columns = [col for col in merged_df.columns if col not in ['Enrollment', 'Name']]
            if date_columns:
                merged_df['Attendance %'] = merged_df[date_columns].mean(axis=1) * 100
                merged_df['Attendance %'] = merged_df['Attendance %'].round(2)
            
            # Save summary
            summary_path = os.path.join(subject_path, "attendance_summary.csv")
            merged_df.to_csv(summary_path, index=False)
            
            # Display in table
            self.show_summary_table(subject, merged_df)
            
        except Exception as e:
            self.speak(f"Error viewing attendance: {str(e)}")
    
    def show_summary_table(self, subject, df):
        """Show attendance summary in a table"""
        summary_window = tk.Toplevel(self.root)
        summary_window.title(f"Attendance Summary - {subject}")
        summary_window.geometry("600x400")
        summary_window.configure(bg="#1c1c1c")
        
        # Create treeview
        columns = list(df.columns)
        tree = ttk.Treeview(summary_window, columns=columns, show='headings')
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)
        
        for _, row in df.iterrows():
            tree.insert('', tk.END, values=list(row))
        
        # Add scrollbars
        v_scrollbar = ttk.Scrollbar(summary_window, orient=tk.VERTICAL, command=tree.yview)
        h_scrollbar = ttk.Scrollbar(summary_window, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        tree.pack(fill=tk.BOTH, expand=True, padx=(10, 25), pady=(10, 25))
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def open_attendance_folder(self, subject):
        """Open the attendance folder for a subject"""
        if not subject:
            self.speak("Please enter subject name")
            return
        
        subject_path = os.path.join(self.attendance_path, subject)
        if os.path.exists(subject_path):
            os.startfile(subject_path)
        else:
            self.speak("No attendance folder found for this subject")
    
    def run(self):
        """Run the application"""
        self.speak("Welcome to Class Vision Attendance System")
        self.root.mainloop()

if __name__ == "__main__":
    app = AttendanceSystem()
    app.run()