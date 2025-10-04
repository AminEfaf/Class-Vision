# Class Vision
A face recognition-based attendance system built with Python and OpenCV. ClassVision provides an intuitive interface for automated student attendance tracking using real-time facial recognition technology.

---

## Project Overview
ClassVision offers a complete attendance management solution designed for:
- **Educational institutions** looking to automate attendance tracking
- **Teachers** needing efficient classroom management tools
- **Administrators** requiring accurate attendance records and analytics
- **Students** benefiting from contactless attendance marking

This project focuses on providing a user-friendly desktop application with robust facial recognition capabilities, automated record keeping, and comprehensive attendance analytics with voice feedback support.

---

## Features

### Core Functionality
1. **Student Registration**
   - Real-time face capture and training data collection
   - Automatic enrollment number and name mapping
   - Support for 100+ image samples per student
   - Live camera preview with face detection feedback

2. **Intelligent Face Recognition**
   - LBPH (Local Binary Patterns Histograms) algorithm
   - High accuracy detection with confidence scoring
   - Real-time face detection using Haar Cascade
   - Multi-face recognition in single frame

3. **Automated Attendance Tracking**
   - One-click attendance marking for entire class
   - Subject-wise attendance organization
   - Automatic duplicate prevention
   - Real-time student recognition feedback
   - Start/Stop controls for flexible attendance sessions

4. **Student Management**
   - Individual student deletion with confirmation
   - Bulk deletion option for all students
   - Automatic training data cleanup
   - Model retraining suggestions after deletions

5. **Comprehensive Record Management**
   - Date-stamped attendance records
   - Subject-wise folder organization
   - Attendance summary generation
   - Percentage calculation for each student
   - Direct folder access for record review

6. **Voice Feedback System**
   - Text-to-speech notifications
   - Real-time status updates
   - Error notifications and confirmations
   - Enhanced user experience with audio feedback

7. **Modern User Interface**
   - Clean, dark-themed design
   - Intuitive button layout
   - Real-time status updates
   - Modal windows for focused tasks
   - Progress indicators and visual feedback

---

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Webcam or external camera
- Windows/Linux/macOS

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ClassVision.git
   cd ClassVision
   ```

2. **Install dependencies**
   ```bash
   pip install opencv-python opencv-contrib-python pillow pandas numpy pyttsx3
   ```

3. **Download Haar Cascade file**
   - The system uses `haarcascade_frontalface_default.xml`
   - Place it in the project root directory
   - Download from OpenCV GitHub repository if not included

4. **Run the application**
   ```bash
   python ClassVision.py
   ```

5. **Initial Setup**
   - The application will automatically create required directories
   - Click "Add Student" to register your first student
   - Train the model after adding students
   - Start taking attendance!

---

## Technical Details

### Face Recognition Technology
- **Algorithm**: LBPH (Local Binary Patterns Histograms) Face Recognizer
- **Detection**: Haar Cascade Classifier for frontal face detection
- **Training**: Custom model training on captured student images
- **Recognition Threshold**: Confidence < 60 for positive identification

### Data Management
**Directory Structure**
- `TrainingImage/`: Stores captured student face images
- `TrainingImageLabel/`: Contains trained recognition model
- `StudentDetails/`: CSV database of student information
- `Attendance/`: Subject-wise attendance records organized by date

**File Formats**
- Student images: JPG format with naming convention `Name_Enrollment_Count.jpg`
- Student database: CSV with Enrollment and Name columns
- Attendance records: CSV with timestamps and presence markers
- Trained model: YML format for OpenCV recognizer

### User Interface
- **Framework**: Tkinter for native desktop GUI
- **Styling**: Custom dark theme with yellow accents
- **Layout**: Responsive design with proper spacing and alignment
- **Threading**: Background processing for attendance to prevent UI freezing

### Audio Feedback
- **Engine**: pyttsx3 for cross-platform text-to-speech
- **Notifications**: Status updates, confirmations, and error messages
- **Languages**: Configurable voice settings

---

## Usage Guide

### Adding Students
1. Click "Add Student" from the main menu
2. Enter student enrollment number and name
3. Click "Capture Images" to start camera
4. Position face in frame and let system capture 100 samples
5. Press 'q' or wait for completion
6. Click "Train Model" to update recognition system

### Taking Attendance
1. Click "Check Roll" from main menu
2. Enter subject name for the session
3. Click "Start Attendance" to begin recognition
4. Students face the camera for automatic marking
5. System announces recognized students via voice
6. Click "Stop Attendance" when session ends
7. Review attendance table showing all present students

### Viewing Records
1. Click "Check Records" from main menu
2. Enter subject name to view
3. Click "View Summary" for attendance statistics
4. Click "Open Folder" to access CSV files directly
5. Export or analyze data as needed

### Deleting Students
1. Click "Delete Student" from main menu
2. Select student from list or choose "Delete ALL"
3. Confirm deletion when prompted
4. System automatically cleans training data
5. Retrain model when suggested for accuracy

---

## Configuration

### Camera Settings
- Default camera: Index 0 (built-in webcam)
- Change camera index in `cv2.VideoCapture(0)` if needed
- Adjust resolution for performance optimization

### Recognition Parameters
- Face detection scale factor: 1.3 (adjustable for accuracy)
- Minimum neighbors: 5 (adjust for detection sensitivity)
- Confidence threshold: 60 (lower = stricter matching)
- Sample count: 100 images per student

### Directory Customization
All paths are configurable in the `AttendanceSystem.__init__()` method:
- Training images directory
- Student details CSV location
- Attendance records path
- Trained model storage location

---

## Features Breakdown

### Advanced Recognition
- Handles multiple faces in single frame
- Real-time confidence scoring display
- Unknown face detection with visual feedback
- Robust performance in varying lighting conditions

### Data Analytics
- Attendance percentage calculation per student
- Multi-date attendance aggregation
- Subject-wise performance tracking
- Exportable CSV reports for further analysis

### User Experience
- Visual feedback with colored bounding boxes (Green: recognized, Red: unknown)
- Real-time student count display during capture
- Voice confirmations for all major actions
- Modal dialogs for focused data entry
- Scrollable lists for large student databases

### Error Handling
- Automatic directory creation
- Missing file detection and user notification
- Camera access error handling
- CSV parsing error recovery
- Thread-safe attendance operations

---

## Future Enhancements

Potential improvements for future versions:
- Cloud storage integration for backup
- Mobile app companion
- Advanced analytics dashboard
- Multi-camera support
- Deep learning models (CNN, FaceNet)
- Authentication and role-based access
- Email/SMS notifications for absences
- Integration with LMS platforms

---

---

## Team Members

- Mohammad Amin Efaf
- Arman Ghorbanpour 
