import os
import sys
import numpy as np
import librosa
import pickle
import tempfile
from pathlib import Path
import warnings
import tensorflow as tf
from tensorflow.keras.models import load_model
import traceback
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import soundfile as sf
import time
import torch
import pygame
from threading import Thread, Lock
import queue
import json
from obstacle_llm import SimpleTokenizer, SimpleObstacleLLM
from PyQt5.QtWidgets import Qlabel


# Suppress warnings
warnings.filterwarnings("ignore")



# Import robot command functionality from robot_llm
try:
    from robot_llm import RobotCommandLLM, RobotTokenizer, generate_response
    robot_module_available = True
except ImportError:
    robot_module_available = False
    print("Warning: Robot LLM module not found. Robot command execution will be simulated.")

# Import the object description LLM for the Questions tab
try:
    from test_llm import ObjectDescriptionLLM, load_object_description_model, generate_object_description
    object_description_module_available = True
    print("Object description LLM module loaded successfully")
except ImportError:
    object_description_module_available = False
    print("Warning: Object description LLM module not found. Object descriptions will be simulated.")

# Constants
MODEL_PATH = "vacuum_robot_speech_model.keras"
METADATA_PATH = "model_metadata.pkl"
ROBOT_MODEL_PATH = "robot_llm_model.pth"
OBJECT_MODEL_PATH = "model_20250310_213748.pth"
SAMPLE_RATE = 22050
DENOISED_DIR = "denoised_output"
os.makedirs(DENOISED_DIR, exist_ok=True)

# Constants for feature extraction
THRESHOLD = 0.0008  # For audio envelope detection
N_FFT = 2048  # FFT window size
HOP_LENGTH = 512  # Hop length for STFT
N_MFCC = 13  # Number of MFCC coefficients

# Constants from robot.py
GRID_SIZE = 20
CELL_SIZE = 35

# Import the robot simulation components
try:
    # Directly import specific components from the robot module
    from robot import House, Robot, UI, DOCKING_STATION, ObstacleType, ObjectDetector,handle_mouse_click
    robot_simulation_available = True
    print("Robot simulation modules loaded successfully")
except ImportError as e:
    robot_simulation_available = False
    print(f"Warning: Robot simulation module not found: {e}. Simulation will be disabled.")
class ClickableLabel(QLabel):
    clicked = pyqtSignal(object)

    def mousePressEvent(self, event):
        pos = event.pos()
        self.clicked.emit(pos)
        super().mousePressEvent(event)
# Mock robot response function as fallback if module is not available
def mock_generate_response(command):
    """Generate a mock response for robot commands"""
    command = command.lower()
    
    if "move to room" in command or "go to room" in command:
        room = command.split("room")[-1].strip()
        return f"EXECUTING: Moving to Room {room}"
    
    elif "clean room" in command:
        room = command.split("room")[-1].strip()
        return f"EXECUTING: Cleaning Room {room}"
    
    elif "and" in command:
        parts = command.split("and")
        responses = []
        for part in parts:
            if part.strip():
                responses.append(mock_generate_response(part.strip()))
        return "\n".join(responses)
    
    elif "go home" in command:
        return "EXECUTING: Returning to charging station"
    
    elif "tell about" in command:
        obj = command.split("tell about")[-1].strip()
        if "the " in obj:
            obj = obj.replace("the ", "")
        return f"EXECUTING: Providing information about {obj}"
    
    else:
        return f"EXECUTING: {command}"

# Mock object description function as fallback if module is not available
def mock_generate_object_description(question):
    """Generate a mock response for object description questions"""
    question = question.lower()
    
    if "what is" in question:
        object_name = question.split("what is")[-1].strip().strip("?")
        return f"The {object_name} is a common household object that the robot can detect and navigate around."
    
    elif "tell me about" in question:
        object_name = question.split("tell me about")[-1].strip().strip("?")
        return f"The {object_name} is an object that the robot has detected in the environment. It has specific characteristics that help the robot identify it."
    
    elif "how does" in question:
        return "The robot uses computer vision and machine learning algorithms to detect and classify objects in its environment."
    
    elif "detected" in question:
        return "The robot has detected various objects in different rooms. The detections are logged and can be queried for more information."
    
    else:
        return f"I understand you're asking about '{question}'. The robot's object recognition system can provide basic information about detected objects."
class StartScreen(QWidget):
    def __init__(self, start_callback):
        super().__init__()
        self.start_callback = start_callback
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Welcome')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.label = QLabel(self)
        pixmap = QPixmap("bg.jpg")
        self.label.setPixmap(pixmap)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setScaledContents(True)

        layout.addWidget(self.label)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.close()
            self.start_callback()

# Robot model loader
class RobotModelLoader:
    def __init__(self, model_path=ROBOT_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def load_model(self):
        """Load the robot command model"""
        if not robot_module_available:
            print("Robot LLM module not available.")
            return False
            
        if not os.path.exists(self.model_path):
            print(f"Robot model file {self.model_path} not found.")
            return False
            
        try:
            # Load the model using the checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Initialize tokenizer
            self.tokenizer = RobotTokenizer()
            self.tokenizer.vocab = checkpoint['vocab']
            self.tokenizer.id2word = checkpoint['id2word']
            
            # Initialize model with the same parameters used during training
            self.model = RobotCommandLLM(
                vocab_size=len(self.tokenizer.vocab),
                embedding_dim=256,  # Match the embedding_dim used in training
                hidden_dim=512,
                num_layers=3,
                dropout=0.3,
                min_length=3
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print("Robot command model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading robot model: {str(e)}")
            traceback.print_exc()
            return False
    
    def execute_command(self, command_text):
        """Execute a command with the robot model"""
        if not self.model or not self.tokenizer:
            return mock_generate_response(command_text)
            
        try:
            response = generate_response(self.model, self.tokenizer, command_text, self.device)
            return response
        except Exception as e:
            print(f"Error executing command: {str(e)}")
            traceback.print_exc()
            return f"Error executing: {command_text}"

# Object Description Model Loader
class ObjectDescriptionModelLoader:
    def __init__(self, model_path=OBJECT_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def load_model(self, model_dir='saved_models'):
        # Load the paths to the latest model and tokenizer
        paths_file = os.path.join(model_dir, 'latest_model_paths.json')
        if not os.path.exists(paths_file):
            print("No saved model found. Please train the model first.")
            return False

        with open(paths_file, 'r') as f:
            paths = json.load(f)

        # Load tokenizer
        with open(paths['tokenizer_path'], 'r') as f:
            vocab = json.load(f)
        self.tokenizer = SimpleTokenizer()
        self.tokenizer.vocab = vocab
        self.tokenizer.id2word = {v: k for k, v in vocab.items()}

        # Load model
        checkpoint = torch.load(paths['model_path'], map_location=self.device)
        self.model = SimpleObstacleLLM(checkpoint['vocab_size'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        return True

        print("Model and tokenizer loaded successfully!")
    
    def get_object_description(self, question_text):
        if self.model is None or self.tokenizer is None:
            return mock_generate_object_description(question_text)

        with torch.no_grad():
            tokens = self.tokenizer.encode(question_text)
            tokens = [self.tokenizer.vocab['<START>']] + tokens
            input_tensor = torch.LongTensor([tokens]).to(self.device)
            output_tokens = self.model.generate(input_tensor)
            return self.tokenizer.decode(output_tokens[0].tolist())

def envelope(samples, sample_rate, threshold):
    """Apply an envelope filter to keep only parts with signal above threshold"""
    import pandas as pd
    mask = []
    samples = pd.Series(samples).apply(np.abs)
    sample_mean = samples.rolling(window=int(sample_rate/10), min_periods=1, center=True).mean()
    for mean in sample_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

def pad_zeros(samples, rate):
    """Pad audio with zeros to make it exactly 1 second long"""
    if len(samples) >= rate:
        # If longer than 1 second, truncate
        return samples[:rate]
    else:
        # If shorter than 1 second, pad with zeros
        pad_length = rate - len(samples)
        return np.pad(samples, (0, pad_length), 'constant')

def extract_features(audio_data, threshold=THRESHOLD, n_fft=N_FFT, 
                     hop_length=HOP_LENGTH, n_mfcc=N_MFCC, rate=SAMPLE_RATE):
    """Extract MFCC features from audio data"""
    try:
        # If audio_data is already a numpy array, use it directly
        if isinstance(audio_data, np.ndarray):
            samples = audio_data
        else:
            # Otherwise, assume it's a file path and load it
            samples, _ = librosa.load(audio_data, sr=rate)
        
        # Remove silence parts
        mask = envelope(samples, rate, threshold)
        samples = samples[mask]
        
        # Ensure the sample is exactly 1 second (pad or truncate)
        samples = pad_zeros(samples, rate)
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=samples, sr=rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        
        # Add channel dimension for CNN
        return np.expand_dims(mfccs, axis=-1)
    
    except Exception as e:
        print(f"Error extracting features: {e}")
        traceback.print_exc()
        return None

def process_audio_file(file_path):
    """Process a single audio file."""
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        features = extract_features(audio)
        return features, audio
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None, None

# Audio waveform visualization widget
class WaveformVisualization(FigureCanvasQTAgg):
    def __init__(self, width=5, height=4, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setMinimumHeight(300)
    
    def plot_waveform(self, audio_data, sr, title="Waveform"):
        """Plot audio waveform"""
        self.ax.clear()
        librosa.display.waveshow(audio_data, sr=sr, ax=self.ax, alpha=0.8)
        self.ax.set_title(title)
        self.ax.set_ylabel("Amplitude")
        self.ax.set_xlabel("Time (s)")
        self.fig.tight_layout()
        self.draw()
    
    def plot_spectrogram(self, audio_data, sr, title="Spectrogram"):
        """Plot audio spectrogram"""
        self.ax.clear()
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=self.ax)
        self.fig.colorbar(self.ax.collections[0], ax=self.ax, format='%+2.0f dB')
        self.ax.set_title(title)
        self.fig.tight_layout()
        self.draw()
    
    def plot_comparison(self, original_audio, denoised_audio, sr, title="Comparison"):
        """Plot original vs denoised waveforms"""
        self.ax.clear()
        time = np.arange(len(original_audio)) / sr
        
        # Ensure same length for display
        min_len = min(len(original_audio), len(denoised_audio))
        time = time[:min_len]
        original_audio = original_audio[:min_len]
        denoised_audio = denoised_audio[:min_len]
        
        self.ax.plot(time, original_audio, alpha=0.7, label="Original")
        self.ax.plot(time, denoised_audio, alpha=0.7, label="Processed")
        self.ax.set_title(title)
        self.ax.set_ylabel("Amplitude")
        self.ax.set_xlabel("Time (s)")
        self.ax.legend()
        self.fig.tight_layout()
        self.draw()
    
    def clear(self):
        """Clear the plot"""
        self.ax.clear()
        self.ax.set_title("No audio loaded")
        self.draw()

# Pygame Surface to QPixmap converter
def pygame_surface_to_qpixmap(surface):
    """Convert a pygame surface to a QPixmap for display in Qt"""
    w, h = surface.get_size()
    
    # Create a copy of the surface with the right format
    surface_copy = pygame.Surface((w, h), pygame.SRCALPHA, 32)
    surface_copy.blit(surface, (0, 0))
    
    # Get the byte representation
    surface_bytes = pygame.image.tostring(surface_copy, "RGBA")
    
    # Create QImage directly from bytes (not from numpy array)
    qimage = QImage(surface_bytes, w, h, QImage.Format_RGBA8888)
    
    # Convert QImage to QPixmap
    return QPixmap.fromImage(qimage)

# Customized Robot class to fix the object detection in all rooms
class CustomRobot(Robot):
    def __init__(self, x, y, house):
        super().__init__(x, y, house)
        self.visited_rooms = set()
        
    def update(self, current_time):
        """Update robot position and check for collisions"""
        # Update current room
        current_room = self.house.get_room_at(self.x, self.y)
        if current_room and current_room not in self.visited_rooms:
            self.visited_rooms.add(current_room)
            print(f"Robot entered room {current_room} for the first time!")
            # Force detection in newly entered room
            self.perform_room_detection(current_room)
            
        # Call the parent update method
        super().update(current_time)
    
    def perform_room_detection(self, room_id=None):
        """Scan for objects in a specific room"""
        # If no room_id provided, use the current target room
        if room_id is None and self.target_room:
            room_id = self.target_room
        
        if not room_id:
            return
            
        print(f"[Detection] Scanning Room {room_id}...")

        room_objects = self.house.obstacle_manager.object_positions.get(room_id, {})
        if not room_objects:
            print(f"[Detection] No objects to detect in Room {room_id}")
            return

        for (obj_x, obj_y), expected_type in room_objects.items():
            # Load a random best image from the correct class folder
            best_image = self.load_random_best_image(expected_type)

            if best_image is None:
                print(f"[Detection] Skipping detection at ({obj_x}, {obj_y}) due to missing image")
                continue

            # Use the detector
            predicted_label, confidence = self.object_detector.detect(best_image)

            # Compare prediction with expected
            expected_label = expected_type.value

            if predicted_label == expected_label:
                print(f"[Success] {expected_label} correctly detected with {confidence:.2f}% confidence in Room {room_id}")
                # Log the detection to file
                self.detection_logger.log_object(room_id, expected_label)
            else:
                print(f"[Mismatch] Expected {expected_label}, but detected {predicted_label} ({confidence:.2f}%) in Room {room_id}")

# Pygame rendering thread to avoid blocking Qt's main loop
class PygameRenderThread(QThread):
    update_frame = pyqtSignal(QPixmap)
    detection_update = pyqtSignal(str)
    
    def __init__(self, command_queue):
        super().__init__()
        self.running = True
        self.command_queue = command_queue
        
        # Initialize pygame
        pygame.init()
        
        #  Create an offscreen surface
        self.width = GRID_SIZE * CELL_SIZE
        self.height = GRID_SIZE * CELL_SIZE + 80  # Include UI
        self.surface = pygame.Surface((self.width, self.height))
        
        # Initialize house, robot, and UI
        self.house = House()
        # Use the customized robot class instead
        self.robot = CustomRobot(DOCKING_STATION[0], DOCKING_STATION[1], self.house)
        self.ui = UI(self.house)
        
        # Clock for frame rate control
        self.clock = pygame.time.Clock()
        
        # Thread lock for pygame calls
        self.lock = Lock()
        
        # Load detections file if it exists, otherwise create it
        self.detections_file = 'detections.txt'
        if not os.path.exists(self.detections_file):
            with open(self.detections_file, 'w') as f:
                f.write("# Robot Detection Log\n\n")
    
    def run(self):
        """Main thread loop"""
        while self.running:
            try:
                events = pygame.event.get()
            except pygame.error:
                break  # Exit the thread gracefully if video system is uninitialized

            self.process_commands()
            for event in events:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    with self.lock:
                        handle_mouse_click(event.pos, self.robot, self.ui.buttons)

            # Update robot state
            current_time = pygame.time.get_ticks()
            with self.lock:
                self.robot.update(current_time)

            # Render the simulation
            self.render_simulation()

            # Read and emit detection updates
            self.check_detections()

            # Control frame rate
            self.clock.tick(30)

    
    def process_commands(self):
        """Process commands from the queue"""
        try:
            while not self.command_queue.empty():
                command = self.command_queue.get_nowait()
                print(f"Processing command: {command}")
                
                command_text = command.lower()
                
                # Handle move to room commands
                if "move to room" in command_text or "go to room" in command_text:
                    room = command_text.split("room")[-1].strip().upper()
                    print(f"Directing robot to Room {room}")
                    with self.lock:
                        self.robot.add_to_queue(room)
                
                # Handle clean room commands
                elif "clean room" in command_text:
                    room = command_text.split("room")[-1].strip().upper()
                    print(f"Directing robot to clean Room {room}")
                    with self.lock:
                        self.robot.add_to_queue(room)
                
                # Handle multi-room commands with 'and'
                elif "and" in command_text:
                    parts = command_text.split("and")
                    for part in parts:
                        if "room" in part:
                            room_part = part.strip()
                            # Add this as a new command to process
                            self.command_queue.put(room_part)
                
                # Handle return to dock commands
                elif "go home" in command_text or "return to dock" in command_text:
                    print("Directing robot to return to dock")
                    with self.lock:
                        self.robot.return_to_dock()
                
                # Handle tell about commands - these don't affect robot movement
                elif "tell about" in command_text:
                    # No robot movement needed, just informational
                    pass
                
                # Mark command as processed
                self.command_queue.task_done()
        except queue.Empty:
            pass
    
    def render_simulation(self):
        """Render the robot simulation to the surface"""
        with self.lock:
            # Draw the house layout
            self.house.draw(self.surface)
            
            # Draw the robot's path and the robot
            self.robot.draw_path(self.surface)
            self.robot.draw(self.surface)
            
            # Draw the UI elements
            self.ui.draw(self.surface, self.robot)
        
        # Convert the pygame surface to a QPixmap and emit the signal
        pixmap = pygame_surface_to_qpixmap(self.surface)
        self.update_frame.emit(pixmap)
    
    def check_detections(self):
        """Check if there are new detections and emit them"""
        try:
            if os.path.exists(self.detections_file):
                with open(self.detections_file, 'r') as f:
                    content = f.read()
                    self.detection_update.emit(content)
        except Exception as e:
            print(f"Error reading detections file: {e}")
    
    def stop(self):
        """Stop the thread and clean up resources"""
        self.running = False
        self.wait()  # Wait until run() fully exits
        with self.lock:
            if pygame.get_init():
                pygame.quit()


# Worker thread for background processing
class ProcessingThread(QThread):
    update_progress = pyqtSignal(int)
    processing_complete = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, input_file, denoise=True, profile="balanced", execute_command=True, robot_model=None):
        super().__init__()
        self.input_file = input_file
        self.denoise = denoise
        self.profile = profile
        self.execute_command = execute_command
        self.robot_model = robot_model
        
    def run(self):
        try:
            result = {}
            self.update_progress.emit(10)
            
            # Step 1: Load the original audio
            original_audio, sr = librosa.load(self.input_file, sr=SAMPLE_RATE)
            result["original_audio"] = original_audio
            result["original_path"] = self.input_file
            self.update_progress.emit(20)
            
            # Step 2: Fake denoising (just create a slight modification for visualization)
            # We're simulating denoising by just creating a copy of the audio with minimal changes
            if self.denoise:
                # Create a slightly modified version of the audio for UI demonstration
                # In a real implementation, this would be actual denoising
                denoised_audio = original_audio.copy()
                
                # Add a small amount of processing to make it visually different
                # This is just for demonstration, not actual denoising
                if self.profile == "voice_clarity":
                    # Slight amplitude boost in mid-frequencies (very simplistic)
                    denoised_audio = denoised_audio * 1.05
                elif self.profile == "noise_reduction":
                    # Simple high-frequency attenuation (not real noise reduction)
                    stft = librosa.stft(denoised_audio)
                    stft[-20:] *= 0.8  # Slightly reduce highest frequencies
                    denoised_audio = librosa.istft(stft)
                
                # Create output filename
                output_filename = os.path.join(DENOISED_DIR, 
                                             f"processed_{os.path.basename(self.input_file)}")
                
                # Save the "processed" audio
                sf.write(output_filename, denoised_audio, SAMPLE_RATE)
                
                result["denoised_audio"] = denoised_audio
                result["denoised_path"] = output_filename
                audio_path_for_recognition = self.input_file  # Use ORIGINAL for recognition
                self.update_progress.emit(40)
            else:
                # Skip denoising
                result["denoised_audio"] = original_audio
                result["denoised_path"] = self.input_file
                audio_path_for_recognition = self.input_file
                self.update_progress.emit(40)
            
            # Step 3: Voice Recognition (use original audio for recognition)
            self.update_progress.emit(60)
            try:
                # Use the process_audio_file function on ORIGINAL audio for reliable results
                features, audio = process_audio_file(audio_path_for_recognition)
                
                # Load model and metadata
                with open(METADATA_PATH, 'rb') as f:
                    metadata = pickle.load(f)
                
                # Load the classification model
                model = load_model(MODEL_PATH)
                
                # Get encoder and mapping for decoding
                inverse_mapping = metadata.get("inverse_mapping")
                
                if features is not None:
                    # Make prediction
                    prediction = model.predict(np.expand_dims(features, axis=0), verbose=0)
                    
                    # Get class with highest probability
                    class_index = np.argmax(prediction[0])
                    confidence = prediction[0][class_index]
                    
                    # Get class name
                    if inverse_mapping:
                        command_text = inverse_mapping[class_index]
                    else:
                        command_text = f"Class {class_index}"
                    
                    result["recognized_text"] = command_text
                    result["confidence"] = confidence
                    
                    # Step 4: Execute robot command if requested
                    if self.execute_command:
                        self.update_progress.emit(80)
                        if self.robot_model:
                            # Use the actual robot model to execute command
                            robot_response = self.robot_model.execute_command(command_text)
                        else:
                            # Use mock response as fallback
                            robot_response = mock_generate_response(command_text)
                            
                        result["robot_response"] = robot_response
                        result["command_text"] = command_text  # Save the original command text
                else:
                    result["recognized_text"] = "Feature extraction failed"
                    result["confidence"] = 0.0
                    result["robot_response"] = "Command execution failed: No valid command recognized"
                
            except Exception as e:
                error_msg = f"Error in voice recognition: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                result["recognized_text"] = "Recognition failed"
                result["confidence"] = 0.0
                result["robot_response"] = "Command execution failed: Recognition error"
                self.error_occurred.emit(error_msg)
            
            self.update_progress.emit(100)
            self.processing_complete.emit(result)
            
        except Exception as e:
            error_msg = f"Error during processing: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            self.error_occurred.emit(error_msg)

# Main application window
class RobotVoiceCommandSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.command_queue = queue.Queue()
        self.initUI()
        self.loadModels()
        
        # Store audio data
        self.original_audio = None
        self.denoised_audio = None
        self.sr = SAMPLE_RATE
        
        # Processing state
        self.is_processing = False
        
        # Initialize and start the Pygame rendering thread if available
        if robot_simulation_available:
            self.pygame_thread = PygameRenderThread(self.command_queue)
            self.pygame_thread.update_frame.connect(self.update_simulation_view)
            self.pygame_thread.detection_update.connect(self.update_detections_tab)
            self.pygame_thread.start()
        else:
            self.pygame_thread = None
            
        self.setStyleSheet("""
            QWidget {
                font-family: Arial;
                font-size: 11pt;
                background-color: #f5f5f5;
            }
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0069d9;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
            QLabel {
                color: #343a40;
            }
            QTextEdit {
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 8px;
                background-color: white;
            }
            QLineEdit {
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 8px;
                background-color: white;
            }
            QGroupBox {
                border: 1px solid #ddd;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 16px;
                font-weight: bold;
            }
            QProgressBar {
                border: 1px solid #bbb;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #007bff;
                width: 10px;
            }
            QTabWidget::pane {
                border: 1px solid #ddd;
                border-radius: 4px;
                top: -1px;
            }
            QTabBar::tab {
                background-color: #f0f0f0;
                border: 1px solid #ddd;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 6px 12px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #ffffff;
                border-bottom: 1px solid #ffffff;
            }
        """)

    def loadModels(self):
        """Load the voice recognition and robot models."""
        try:
            # Check if robot simulation is available
            if not robot_simulation_available:
                self.simulation_frame.setText("Robot simulation not available.\nMake sure robot.py is in the same directory.")
                self.simulation_status.setText("Simulation disabled")
            
            # Check and load voice recognition model
            if not os.path.exists(MODEL_PATH):
                # Try alternate extension
                alt_path = MODEL_PATH.replace('.keras', '.h5')
                if os.path.exists(alt_path):
                    self.model_path = alt_path
                else:
                    self.result_text.setText("Voice recognition model file not found. Please train the model first.")
                    self.process_button.setEnabled(False)
                    return
            else:
                self.model_path = MODEL_PATH
                    
            # Check if the metadata file exists
            if not os.path.exists(METADATA_PATH):
                self.result_text.setText("Metadata file not found. Please run preprocessing first.")
                self.process_button.setEnabled(False)
                return
            
            # Load metadata
            with open(METADATA_PATH, 'rb') as f:
                self.metadata = pickle.load(f)
            
            # Store encoder and mapping for decoding
            self.encoder = self.metadata.get("encoder")
            self.inverse_mapping = self.metadata.get("inverse_mapping")
            
            # Load the robot model
            self.robot_model = RobotModelLoader(ROBOT_MODEL_PATH)
            robot_model_loaded = self.robot_model.load_model()
            
            # Load the object description model
            self.object_model = ObjectDescriptionModelLoader(OBJECT_MODEL_PATH)
            object_model_loaded = self.object_model.load_model()
            
            status_parts = []
            
            if robot_model_loaded:
                status_parts.append("Voice recognition and robot command models loaded successfully.")
            else:
                status_parts.append("Voice recognition model loaded. Robot command model not available - using simulated responses.")
            
            if object_model_loaded:
                status_parts.append("Object description model loaded successfully.")
            else:
                status_parts.append("Object description model not available - using simulated responses.")
                
            if robot_simulation_available:
                status_parts.append("Robot simulation active.")
            else:
                status_parts.append("Robot simulation not available.")
                
            self.status_display.setText("\n".join(status_parts))
            
        except Exception as e:
            self.result_text.setText(f"Error loading models: {str(e)}")
            traceback.print_exc()
            self.process_button.setEnabled(False)

    def initUI(self):
        self.setWindowTitle('Conversational AI based Robot Vacuum Cleaner')
        self.setGeometry(100, 100, 1600, 900)  # Wider window to accommodate both panels

        # Main layout with horizontal splitter
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)

        # Header
        header_label = QLabel('Conversational AI based Robot Vacuum Cleaner')
        header_label.setStyleSheet('font-size: 20px; font-weight: bold; margin: 10px;')
        header_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header_label)

        
        
        # Create a horizontal splitter for left (original UI) and right (robot simulation) panels
        self.main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.main_splitter, 1)  # Stretch to fill available space
        
        # === Create left panel (original integrated.py UI) ===
        self.left_panel = QWidget()
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)  # Smaller margins
        
        # Create a vertical splitter for controls and results
        self.left_splitter = QSplitter(Qt.Vertical)
        left_layout.addWidget(self.left_splitter)
        
        # Upper left panel - controls
        left_controls = QWidget()
        left_controls_layout = QVBoxLayout(left_controls)
        left_controls_layout.setContentsMargins(5, 5, 5, 5)  # Smaller margins
        left_controls_layout.setSpacing(5)  # Reduced spacing
        
        # File selection section
        file_group = QGroupBox("Audio Input")
        file_layout = QVBoxLayout(file_group)
        file_layout.setContentsMargins(5, 15, 5, 5)  # Smaller margins, keep top margin for title
        file_layout.setSpacing(5)  # Reduced spacing
        
        # File selection row
        file_row = QWidget()
        file_row_layout = QHBoxLayout(file_row)
        file_row_layout.setContentsMargins(0, 0, 0, 0)
        
        self.file_label = QLabel('No file selected')
        self.file_button = QPushButton('Select Audio File')
        self.file_button.setMinimumHeight(30)
        file_row_layout.addWidget(self.file_label)
        file_row_layout.addWidget(self.file_button)
        file_layout.addWidget(file_row)
        
        left_controls_layout.addWidget(file_group)
        
        # Processing parameters section (formerly denoising)
        process_group = QGroupBox("Processing Settings")
        process_layout = QVBoxLayout(process_group)
        process_layout.setContentsMargins(5, 15, 5, 5)  # Smaller margins, keep top margin for title
        process_layout.setSpacing(5)  # Reduced spacing

        # Make processing settings more horizontal to save vertical space
        process_row1 = QWidget()
        process_row1_layout = QHBoxLayout(process_row1)
        process_row1_layout.setContentsMargins(0, 0, 0, 0)
        
        self.process_checkbox = QCheckBox("Apply Audio Processing")
        self.process_checkbox.setChecked(True)
        process_layout.addWidget(self.process_checkbox)
        
        profile_row = QWidget()
        profile_layout = QHBoxLayout(profile_row)
        profile_layout.setContentsMargins(0, 0, 0, 0)
        
        profile_label = QLabel("Processing Profile:")
        self.profile_combo = QComboBox()
        self.profile_combo.addItems(["balanced", "voice_clarity", "noise_reduction", "conference", "podcast"])
        
        profile_layout.addWidget(profile_label)
        profile_layout.addWidget(self.profile_combo)
        process_layout.addWidget(profile_row)
        
        # Robot command execution checkbox
        self.robot_execute_checkbox = QCheckBox("Execute Robot Commands")
        self.robot_execute_checkbox.setChecked(True)
        process_layout.addWidget(self.robot_execute_checkbox)
        
        left_controls_layout.addWidget(process_group)
        
        # Visualization settings
        visual_group = QGroupBox("Visualization")
        visual_layout = QHBoxLayout(visual_group)
        visual_layout.setContentsMargins(5, 15, 5, 5)  # Smaller margins, keep top margin for title
        
        viz_row = QWidget()
        viz_layout = QHBoxLayout(viz_row)
        viz_layout.setContentsMargins(0, 0, 0, 0)
        
        self.viz_type_group = QButtonGroup(self)
        self.viz_waveform_radio = QRadioButton("Waveform")
        self.viz_spectrogram_radio = QRadioButton("Spectrogram")
        self.viz_comparison_radio = QRadioButton("Comparison")
        
        self.viz_type_group.addButton(self.viz_waveform_radio)
        self.viz_type_group.addButton(self.viz_spectrogram_radio)
        self.viz_type_group.addButton(self.viz_comparison_radio)
        
        self.viz_waveform_radio.setChecked(True)
        
        viz_layout.addWidget(self.viz_waveform_radio)
        viz_layout.addWidget(self.viz_spectrogram_radio)
        viz_layout.addWidget(self.viz_comparison_radio)
        visual_layout.addWidget(viz_row)
        
        left_controls_layout.addWidget(visual_group)
        
        # Process button
        self.process_button = QPushButton('Process and Execute')
        self.process_button.setMinimumHeight(40)
        self.process_button.setEnabled(False)
        left_controls_layout.addWidget(self.process_button)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        left_controls_layout.addWidget(self.progress_bar)
        
        # Status display
        self.status_display = QTextEdit()
        self.status_display.setReadOnly(True)
        self.status_display.setMaximumHeight(80)
        self.status_display.setPlaceholderText("Status messages will appear here")
        left_controls_layout.addWidget(self.status_display)
        
        # Add the controls to the left splitter
        self.left_splitter.addWidget(left_controls)
        
        # Lower left panel - results
        left_results = QWidget()
        left_results_layout = QVBoxLayout(left_results)
        
        # Create tab widget for different result views
        self.tab_widget = QTabWidget()
        
        # Tab 1: Audio Visualization
        self.audio_viz_tab = QWidget()
        audio_viz_layout = QVBoxLayout(self.audio_viz_tab)
        audio_viz_layout.setContentsMargins(5, 5, 5, 5)  # Smaller margins
        
        viz_label = QLabel("Audio Visualization")
        viz_label.setFont(QFont('Arial', 11, QFont.Bold))
        viz_label.setAlignment(Qt.AlignCenter)
        audio_viz_layout.addWidget(viz_label)
        
        self.waveform_viz = WaveformVisualization(width=5, height=3.5, dpi=100)
        audio_viz_layout.addWidget(self.waveform_viz)
        
        # Tab 2: Recognition Results
        self.recog_tab = QWidget()
        recog_layout = QVBoxLayout(self.recog_tab)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(300)
        self.result_text.setPlaceholderText("Voice recognition results will appear here")
        recog_layout.addWidget(self.result_text)
        
        # Tab 3: Robot Execution
        self.robot_tab = QWidget()
        robot_layout = QVBoxLayout(self.robot_tab)
        
        self.robot_text = QTextEdit()
        self.robot_text.setReadOnly(True)
        self.robot_text.setMinimumHeight(300)
        self.robot_text.setPlaceholderText("Robot command execution results will appear here")
        robot_layout.addWidget(self.robot_text)
        
        # Tab 4: Detections (was Examples)
        self.detections_tab = QWidget()
        detections_layout = QVBoxLayout(self.detections_tab)
        
        self.detections_text = QTextEdit()
        self.detections_text.setReadOnly(True)
        self.detections_text.setMinimumHeight(300)
        self.detections_text.setPlaceholderText("Object detection results will appear here")
        detections_layout.addWidget(self.detections_text)
        
        # Tab 5: Questions - NEW TAB
        self.questions_tab = QWidget()
        questions_layout = QVBoxLayout(self.questions_tab)
        
        # Add instruction label for questions tab
        question_instructions = QLabel("Ask questions about objects the robot has detected")
        question_instructions.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        questions_layout.addWidget(question_instructions)
        
        # Add question input area
        question_input_layout = QHBoxLayout()
        
        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText("Type your question here (e.g., 'What is a shoe?' or 'Tell me about dust pan')")
        self.question_input.returnPressed.connect(self.handle_question)
        
        self.ask_button = QPushButton("Ask")
        self.ask_button.clicked.connect(self.handle_question)
        
        question_input_layout.addWidget(self.question_input, 4)  # 80% width
        question_input_layout.addWidget(self.ask_button, 1)      # 20% width
        
        questions_layout.addLayout(question_input_layout)
        
        # Conversation history display
        conversation_label = QLabel("Conversation History")
        conversation_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        questions_layout.addWidget(conversation_label)
        
        self.conversation_display = QTextEdit()
        self.conversation_display.setReadOnly(True)
        self.conversation_display.setMinimumHeight(250)
        self.conversation_display.setPlaceholderText("Your conversation with the robot will appear here")
        self.conversation_display.setStyleSheet("""
            QTextEdit {
                background-color: white;
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 10px;
                font-family: Arial;
                font-size: 11pt;
            }
        """)
        questions_layout.addWidget(self.conversation_display)
        
        # Add tabs to tab widget
        self.tab_widget.addTab(self.audio_viz_tab, "Audio")
        self.tab_widget.addTab(self.recog_tab, "Recognition")
        self.tab_widget.addTab(self.robot_tab, "Robot Execution")
        self.tab_widget.addTab(self.detections_tab, "Detections")
        self.tab_widget.addTab(self.questions_tab, "Questions")
        
        left_results_layout.addWidget(self.tab_widget)
        
        # Add the results to the left splitter
        self.left_splitter.addWidget(left_results)
        
        # Set the initial left splitter proportions
        self.left_splitter.setSizes([400, 600])
        
        # Add the left panel to the main splitter
        self.main_splitter.addWidget(self.left_panel)
        
        # === Create right panel (robot simulation) ===
        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)
        
        # Simulation title
        simulation_label = QLabel("Robot Simulation")
        simulation_label.setFont(QFont('Arial', 16, QFont.Bold))
        simulation_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(simulation_label)
        
        # Frame to display the pygame simulation
        self.simulation_frame = ClickableLabel()
        self.simulation_frame.clicked.connect(self.handle_simulation_click)
        self.simulation_frame.setMinimumSize(GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE + 80)
        self.simulation_frame.setAlignment(Qt.AlignCenter)
        self.simulation_frame.setStyleSheet("background-color: black; border: 2px solid #555;")
        self.simulation_frame.setText("Robot simulation loading...")
        right_layout.addWidget(self.simulation_frame)
        
        # Status bar for simulation
        self.simulation_status = QLabel("Waiting for commands...")
        self.simulation_status.setAlignment(Qt.AlignCenter)
        self.simulation_status.setStyleSheet("font-style: italic; color: #555;")
        right_layout.addWidget(self.simulation_status)
        
        # Add the right panel to the main splitter
        self.main_splitter.addWidget(self.right_panel)
        
        # Set the initial main splitter proportions
        self.main_splitter.setSizes([800, 800])
        
        # Setup connections
        self.file_button.clicked.connect(self.selectFile)
        self.process_button.clicked.connect(self.processAudio)
        
        # Visualization radio buttons
        self.viz_waveform_radio.toggled.connect(self.updateVisualization)
        self.viz_spectrogram_radio.toggled.connect(self.updateVisualization)
        self.viz_comparison_radio.toggled.connect(self.updateVisualization)
        
        # Processing checkbox
        self.process_checkbox.toggled.connect(self.updateProcessSettings)
    def handle_simulation_click(self, pos):
        """Handle clicks inside the simulation QLabel, correctly mapping to Pygame coordinates."""
        if self.pygame_thread:
            label_width = self.simulation_frame.width()
            label_height = self.simulation_frame.height()

            surface_width = self.pygame_thread.width
            surface_height = self.pygame_thread.height

            # Calculate scaling factors
            scale_x = surface_width / label_width
            scale_y = surface_height / label_height

            # Scale click position
            x = int(pos.x() * scale_x)
            y = int(pos.y() * scale_y)

            # Decide if the click is on the grid (map) or UI area
            if y < surface_height - 80:
                # Click is on the simulation grid
                scaled_pos = (x, y)
            else:
                # Click is on the UI area
                # UI expects clicks in **surface** coordinates directly (no reverse scaling needed)
                scaled_pos = (x, y)

            # Pass the correct scaled click position
            with self.pygame_thread.lock:
                handle_mouse_click(scaled_pos, self.pygame_thread.robot, self.pygame_thread.ui.buttons)


    def handle_question(self):
        """Process a question from the Questions tab"""
        question = self.question_input.text().strip()
        if not question:
            return
            
        # Clear input field
        self.question_input.clear()
        
        # Add question to conversation history with styling
        self.conversation_display.append(f'<div style="color: #007bff; font-weight: bold;">You: {question}</div>')
        
        # Process the question
        self.process_question(question)
    
    def process_question(self, question):
        """Process a question and generate a response using the object description model"""
        try:
            # Show processing indicator
            self.ask_button.setEnabled(False)
            self.conversation_display.append('<div style="color: #6c757d; font-style: italic;">Robot thinking...</div>')
            
            # Use the object description model to generate a response
            if hasattr(self, 'object_model'):
                response = self.object_model.get_object_description(question)
            else:
                response = mock_generate_object_description(question)
            
            # Append the response to the conversation history with styling
            self.conversation_display.append(f'<div style="color: #28a745; font-weight: bold;">Robot: </div><div style="margin-left: 20px;">{response}</div>')
            
            # Scroll to the bottom of the conversation
            self.conversation_display.verticalScrollBar().setValue(
                self.conversation_display.verticalScrollBar().maximum()
            )
            
            # Re-enable the ask button
            self.ask_button.setEnabled(True)
            
            # Switch to the Questions tab
            self.tab_widget.setCurrentIndex(4)
            
        except Exception as e:
            # Handle any errors
            error_msg = f"Error processing question: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            self.conversation_display.append(f'<div style="color: #dc3545;">Error: {error_msg}</div>')
            self.ask_button.setEnabled(True)
    
    def selectFile(self):
        """Open file dialog to select an audio file"""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("WAV Files (*.wav)")
        
        if file_dialog.exec_():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                self.selected_file = file_paths[0]
                self.file_label.setText(f'Selected: {os.path.basename(self.selected_file)}')
                self.process_button.setEnabled(True)
                
                # Load and display the waveform
                try:
                    self.original_audio, self.sr = librosa.load(self.selected_file, sr=SAMPLE_RATE)
                    self.waveform_viz.plot_waveform(self.original_audio, self.sr, 
                                                   f"Waveform: {os.path.basename(self.selected_file)}")
                    
                    # Reset denoised audio
                    self.denoised_audio = None
                    
                    # Switch to the audio tab
                    self.tab_widget.setCurrentIndex(0)
                    
                    # Update status
                    self.status_display.setText(f"Loaded audio file: {os.path.basename(self.selected_file)}\n"
                                              f"Duration: {len(self.original_audio)/self.sr:.2f} seconds\n"
                                              f"Sample rate: {self.sr} Hz\n"
                                              f"Click 'Process and Execute' to recognize and execute command.")
                    
                    # Clear result displays
                    self.result_text.clear()
                    self.robot_text.clear()
                    
                except Exception as e:
                    self.status_display.setText(f"Error loading audio: {str(e)}")
                    traceback.print_exc()
    
    def updateProcessSettings(self):
        """Enable or disable processing settings based on checkbox"""
        enabled = self.process_checkbox.isChecked()
        self.profile_combo.setEnabled(enabled)
    
    def update_simulation_view(self, pixmap):
        """Update the simulation view with the latest frame from Pygame"""
        if not self.simulation_frame:
            return
            
        self.simulation_frame.setPixmap(pixmap)
    
    def update_detections_tab(self, content):
        """Update the detections tab with the latest content from detections.txt"""
        if content != self.detections_text.toPlainText():
            self.detections_text.setText(content)
            # Switch to the detections tab if there are new detections
            if "Room" in content and len(content.strip()) > 20:
                self.tab_widget.setCurrentIndex(3)
    
    def send_command_to_robot(self, command_text):
        """Send a command to the robot simulation thread"""
        if self.pygame_thread and command_text:
            # Update status
            self.simulation_status.setText(f"Sending command: {command_text}")
            
            # Add command to queue for the simulation thread
            self.command_queue.put(command_text)
    
    def processAudio(self):
        """Process the selected audio file"""
        if self.is_processing:
            self.status_display.setText("Processing already in progress...")
            return
        
        if not hasattr(self, 'selected_file') or not self.selected_file:
            self.status_display.setText("Please select an audio file first.")
            return
        
        # Update UI state
        self.is_processing = True
        self.process_button.setEnabled(False)
        self.file_button.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # Get settings
        apply_processing = self.process_checkbox.isChecked()
        profile = self.profile_combo.currentText()
        execute_command = self.robot_execute_checkbox.isChecked()
        
        # Update status
        status_text = f"Processing audio file: {os.path.basename(self.selected_file)}\n" \
                     f"Audio processing: {'Enabled' if apply_processing else 'Disabled'}\n" \
                     f"Profile: {profile}\n" \
                     f"Robot command execution: {'Enabled' if execute_command else 'Disabled'}"
        self.status_display.setText(status_text)
        
        # Create processing thread
        self.worker = ProcessingThread(
            input_file=self.selected_file,
            denoise=apply_processing,
            profile=profile,
            execute_command=execute_command,
            robot_model=self.robot_model if hasattr(self, 'robot_model') else None
        )
        
        # Connect signals
        self.worker.update_progress.connect(self.updateProgress)
        self.worker.processing_complete.connect(self.processingComplete)
        self.worker.error_occurred.connect(self.processingError)
        
        # Start processing
        self.worker.start()
    
    def updateProgress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    def processingComplete(self, result):
        """Handle processing completion"""
        # Update UI state
        self.is_processing = False
        self.process_button.setEnabled(True)
        self.file_button.setEnabled(True)
        self.progress_bar.setValue(100)
        
        # Store results
        self.original_audio = result["original_audio"]
        self.denoised_audio = result["denoised_audio"]
        
        # Update visualization
        self.updateVisualization()
        
        # Display recognition results
        recognized_text = result.get("recognized_text", "Recognition failed")
        confidence = result.get("confidence", 0.0)
        
        result_html = f"""
        <h3>Recognition Results:</h3>
        <p><b>Raw Recognition:</b> '{recognized_text}'</p>
        <p><b>Confidence:</b> {confidence:.2f}</p>
        <p><b>Audio File:</b> {os.path.basename(result["original_path"])}</p>
        """
        
        if "denoised_path" in result and result["denoised_path"] != result["original_path"]:
            result_html += f"""
            <p><b>Processed File:</b> {os.path.basename(result["denoised_path"])}</p>
            <p><i>Processed audio saved to: {result["denoised_path"]}</i></p>
            """
        
        self.result_text.setHtml(result_html)
        
        # Display robot execution results
        if "robot_response" in result:
            robot_response = result["robot_response"]
            command_text = result.get("command_text", recognized_text)
           
            # Format robot response with colors
          # Precompute to avoid backslash in f-string
            formatted_response = robot_response.replace('\n', '<br>')



            
            robot_html = f"""
            <h3>Robot Command Execution:</h3>
            <p><b>Command:</b> '{recognized_text}'</p>
            <div style="background-color: #e8f4ff; border-left: 4px solid #0275d8; padding: 10px; margin: 10px 0;">
                <p style="font-family: 'Courier New', monospace; font-size: 14px; color: #0275d8;">
                
                </p>
            </div>
            <p><i>Command executed at: {time.strftime("%H:%M:%S")}</i></p>
            """
            
            self.robot_text.setHtml(robot_html)
            
            # Send the command to the robot simulation if available
            if robot_simulation_available and self.pygame_thread:
                # Send the command to robot simulation thread
                self.send_command_to_robot(command_text)
            
            # Switch to the robot tab to show execution results
            self.tab_widget.setCurrentIndex(2)
        else:
            self.robot_text.setHtml("<p>No robot command was executed.</p>")
        
        # Update status
        self.status_display.setText(f"Processing complete. Voice recognition and command execution successful.\n"
                                  f"Recognized text: '{recognized_text}'\n"
                                  f"Confidence: {confidence:.2f}")
    
    def processingError(self, error_msg):
        """Handle processing errors"""
        # Update UI state
        self.is_processing = False
        self.process_button.setEnabled(True)
        self.file_button.setEnabled(True)
        
        # Update status
        self.status_display.setText(f"Error during processing: {error_msg}")
    
    def updateVisualization(self):
        """Update the visualization based on selected visualization type"""
        if not hasattr(self, 'original_audio') or self.original_audio is None:
            return
        
        # Get the selected visualization type
        if self.viz_waveform_radio.isChecked():
            # Display waveform of active audio (original if no denoised, denoised if available)
            audio_to_show = self.denoised_audio if self.denoised_audio is not None else self.original_audio
            title = "Processed Waveform" if self.denoised_audio is not None else "Original Waveform"
            self.waveform_viz.plot_waveform(audio_to_show, self.sr, title)
            
        elif self.viz_spectrogram_radio.isChecked():
            # Display spectrogram of active audio
            audio_to_show = self.denoised_audio if self.denoised_audio is not None else self.original_audio
            title = "Processed Spectrogram" if self.denoised_audio is not None else "Original Spectrogram"
            self.waveform_viz.plot_spectrogram(audio_to_show, self.sr, title)
            
        elif self.viz_comparison_radio.isChecked():
            # Show comparison if denoised audio is available
            if self.denoised_audio is not None:
                self.waveform_viz.plot_comparison(self.original_audio, self.denoised_audio, 
                                                self.sr, "Original vs Processed")
            else:
                self.waveform_viz.plot_waveform(self.original_audio, self.sr, "Original Waveform")
                self.status_display.setText("Processed audio not available for comparison. "
                                         "Process the audio with processing enabled first.")
    
    def closeEvent(self, event):
        """Clean up resources when closing the application."""
        # Make sure any worker threads are stopped
        if hasattr(self, 'worker') and self.worker is not None and self.worker.isRunning():
            self.worker.wait()
        
        # Stop and clean up pygame rendering thread if it exists
        if hasattr(self, 'pygame_thread') and self.pygame_thread is not None:
            print("Stopping pygame rendering thread...")
            self.pygame_thread.stop()
                
        event.accept()

def main():
    # Check for TensorFlow
    if not tf.__version__:
        print("TensorFlow not properly installed. Please install TensorFlow.")
        return

    app = QApplication(sys.argv)
    global main_window  # <<< ADD THIS
    def launch_robot_system():
        global main_window  # <<< ADD THIS
        main_window = RobotVoiceCommandSystem()
        main_window.showFullScreen()

    start_screen = StartScreen(launch_robot_system)
    start_screen.showFullScreen()

    sys.exit(app.exec_())

if __name__ == "__main__":

    main()




