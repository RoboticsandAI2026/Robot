import pygame
import numpy as np
import math
import time
import random
from queue import PriorityQueue
from enum import Enum
from object_detector import ObjectDetector
import os
from PIL import Image

BEST_DIR = 'best' 
# Initialize Pygame at the very beginning
pygame.init()

# Constants
GRID_SIZE = 20
CELL_SIZE = 35  # Slightly larger cells for better visuals
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE + 80  # Increased UI space
UI_HEIGHT = 80  # Taller UI bar for better spacing
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)

# Room colors - completely different palette
ROOM_A_COLOR = (230, 190, 180)  # Soft terracotta
ROOM_B_COLOR = (180, 220, 180)  # Sage green
ROOM_C_COLOR = (190, 190, 240)  # Lavender
ROOM_D_COLOR = (240, 230, 180)  # Cream

UI_BG = (40, 40, 45)  # Dark background for UI

# Define docking station position
DOCKING_STATION = (3, 17)  # Bottom of Room A

# Setup the window
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Robot Vacuum House Navigation")
clock = pygame.time.Clock()

# Room layout
ROOMS = {
    # Bottom left room
    'A': {'coords': (2, 11, 9, 8), 'color': ROOM_A_COLOR, 'center': (6, 14), 'name': 'A'},
    
    # Top left room
    'B': {'coords': (2, 2, 9, 8), 'color': ROOM_B_COLOR, 'center': (6, 5), 'name': 'B'},
    
    # Top right room
    'C': {'coords': (12, 2, 7, 8), 'color': ROOM_C_COLOR, 'center': (15, 5), 'name': 'C'},
    
    # Middle right room
    'D': {'coords': (12, 11, 7, 8), 'color': ROOM_D_COLOR, 'center': (15, 12), 'name': 'D'},
}

# Wall segments as (x1, y1, x2, y2)
WALL_LINES = [
    # Outer walls
    (1, 1, 19, 1),       # Top
    (1, 1, 1, 19),       # Left
    (1, 19, 19, 19),     # Bottom
    (19, 1, 19, 19),     # Right
    
    # Inner walls - horizontal
    (1, 10, 19, 10),     # Separating top and bottom sections (with gaps for stairs)
    
    # Inner walls - vertical
    (11, 1, 11, 10),     # Separating study and bedroom
    (11, 10, 11, 19),    # Central vertical wall (with gaps)
]

# Doorways/passages - (x, y, width, height)
DOORS = [
    (2, 10, 2, 1),      # Stairs from study to main hall
    (16, 10, 2, 1),     # Stairs from bedroom to main hall
    (11, 16, 1, 2),     # Door from main hall to bathroom
    (11, 8, 1, 2),      # Door between study and bedroom
]

# Create a grid representation of the environment
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

def initialize_grid():
    """Initialize the grid with walls"""
    # Set all cells to free space initially
    grid.fill(0)
    
    # Set the outer bounds of the simulation as walls
    for x in range(GRID_SIZE):
        grid[0][x] = 1      # Top edge of simulation
        grid[GRID_SIZE-1][x] = 1  # Bottom edge of simulation
    for y in range(GRID_SIZE):
        grid[y][0] = 1      # Left edge of simulation
        grid[y][GRID_SIZE-1] = 1  # Right edge of simulation
    
    # Set the walls of the house structure only - not inside the rooms
    # Top and bottom walls
    for x in range(1, 19):
        grid[1][x] = 1      # Top wall
        grid[19][x] = 1     # Bottom wall
    
    # Left and right walls
    for y in range(1, 19):
        grid[y][1] = 1      # Left wall
        grid[y][19] = 1     # Right wall
    
    # Horizontal divider between top and bottom rooms
    for x in range(1, 19):
        if not (2 <= x <= 3 or 16 <= x <= 17):  # Leave gaps for stairs
            grid[10][x] = 1
    
    # Vertical divider between left and right rooms
    for y in range(1, 10):
        if not (8 <= y <= 9):  # Leave gap for door
            grid[y][11] = 1
    
    # Central vertical wall
    for y in range(10, 19):
        if not (16 <= y <= 17):  # Leave gap for bathroom door
            grid[y][11] = 1

def print_grid_debug():
    """Print the grid state for debugging"""
    for y in range(GRID_SIZE):
        row = ""
        for x in range(GRID_SIZE):
            if x == 0 or x == GRID_SIZE-1 or y == 0 or y == GRID_SIZE-1:
                row += "O"  # Outer bounds
            elif grid[y][x] == 1:
                row += "X"  # Wall
            else:
                # Check which room this cell belongs to
                room_found = False
                for room_id, room_data in ROOMS.items():
                    rx, ry, rw, rh = room_data['coords']
                    if rx <= x < rx+rw and ry <= y < ry+rh:
                        row += room_id
                        room_found = True
                        break
                
                if not room_found:
                    row += "."  # Empty space
        print(row)

class ObstacleType(Enum):
    CAP = "cap"
    DUSTPAN = "dust_pan"
    FOOTBALL = "football"
    SHOE = "shoe"
    BOTTLE = "water_bottle"

class Obstacle(pygame.sprite.Sprite):
    def __init__(self, x, y, obstacle_type):
        super().__init__()
        self.type = obstacle_type
        self.grid_x = x
        self.grid_y = y
        
        # Load the image for this obstacle type
        try:
            print(f"Attempting to load image: {obstacle_type.value}.png")
            self.image = pygame.image.load(f"{obstacle_type.value}.png")
            # Scale the image to fit the cell size
            self.image = pygame.transform.scale(self.image, (CELL_SIZE - 4, CELL_SIZE - 4))
            print(f"Successfully loaded image for {obstacle_type.value}")
        except Exception as e:
            print(f"Error loading image for {obstacle_type.value}: {str(e)}")
            # Fallback to drawing if image not found
            self.image = pygame.Surface((CELL_SIZE - 4, CELL_SIZE - 4))
            self.image.fill((255, 0, 0))  # Red color to indicate missing image
            font = pygame.font.SysFont("Arial", 12)
            text = font.render(obstacle_type.value, True, (255, 255, 255))
            text_rect = text.get_rect(center=(CELL_SIZE//2 - 2, CELL_SIZE//2 - 2))
            self.image.blit(text, text_rect)
        
        # Add a colored border to help with debugging
        border_color = (0, 255, 0)  # Green border
        pygame.draw.rect(self.image, border_color, self.image.get_rect(), 2)
        
        self.rect = self.image.get_rect()
        
        # Set position
        self.rect.x = x * CELL_SIZE + 2
        self.rect.y = y * CELL_SIZE + 2

class ObstacleManager:
    def __init__(self, house):
        self.house = house
        self.obstacles = pygame.sprite.Group()
        self.obstacle_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        
        # Define specific positions for each object (one per room)
        self.object_positions = {
            'A': {  # Room A
                (4, 13): ObstacleType.CAP  # Baseball cap in Room A
            },
            'B': {  # Room B
                (4, 7): ObstacleType.DUSTPAN,  # Dust pan in Room B,
                (4, 5): ObstacleType.SHOE  
            },
            'C': {  # Room C
                (15, 4) : ObstacleType.FOOTBALL
            },
            'D': {  # Room D
                (14, 13): ObstacleType.BOTTLE  # Water bottle in Room D
            }
        }
        
        # Add football in Room A (second object in same room)
        
        self.place_objects()
    
    def place_objects(self):
        """Place objects in their predefined positions"""
        for room_id, positions in self.object_positions.items():
            for (x, y), obj_type in positions.items():
                obstacle = Obstacle(x, y, obj_type)
                self.obstacles.add(obstacle)
                self.obstacle_grid[y][x] = 1
    
    def draw(self, surface):
        """Draw all obstacles"""
        self.obstacles.draw(surface)
    
    def get_obstacle_positions(self):
        """Return list of obstacle positions"""
        return [(obs.grid_x, obs.grid_y) for obs in self.obstacles]

class Room:
    def __init__(self, name, coords, color):
        self.name = name
        self.x, self.y, self.width, self.height = coords
        self.color = color
        self.center = (self.x + self.width//2, self.y + self.height//2)
        self.coords = coords  # Store the original coords tuple for easy access
        
    def contains_point(self, x, y):
        return (self.x <= x < self.x + self.width and 
                self.y <= y < self.y + self.height)
    
    def draw(self, surface):
        pygame.draw.rect(surface, self.color,
                        (self.x * CELL_SIZE - 1, 
                         self.y * CELL_SIZE - 1,
                         self.width * CELL_SIZE + 2, 
                         self.height * CELL_SIZE + 2))
        
        # Draw room label
        font = pygame.font.SysFont("Arial", 24, bold=True)
        label = font.render(self.name, True, BLACK)
        label_rect = label.get_rect(
            center=(self.x * CELL_SIZE + (self.width * CELL_SIZE) // 2,
                   self.y * CELL_SIZE + (self.height * CELL_SIZE) // 2)
        )
        surface.blit(label, label_rect)

    def get_coords(self):
        """Return room coordinates in (x, y, width, height) format"""
        return (self.x, self.y, self.width, self.height)

class House:
    def __init__(self):
        self.rooms = {
            'A': Room('A', (2, 11, 9, 8), (230, 190, 180)),  # Soft terracotta
            'B': Room('B', (2, 2, 9, 8), (180, 220, 180)),   # Sage green
            'C': Room('C', (12, 2, 7, 8), (190, 190, 240)),  # Lavender
            'D': Room('D', (12, 11, 7, 8), (240, 230, 180))  # Cream
        }
        
        self.docking_station = (3, 17)
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.initialize_grid()
        self.obstacle_manager = ObstacleManager(self)
    
    def initialize_grid(self):
        """Initialize the grid with walls and room boundaries"""
        # Set outer walls
        self.grid[0, :] = self.grid[-1, :] = 1
        self.grid[:, 0] = self.grid[:, -1] = 1
        
        # Set inner walls
        for x in range(1, 19):
            if not (2 <= x <= 3 or 16 <= x <= 17):
                self.grid[10, x] = 1  # Horizontal divider
        
        # Vertical dividers
        for y in range(1, 10):
            if not (8 <= y <= 9):
                self.grid[y, 11] = 1
        
        for y in range(10, 19):
            if not (16 <= y <= 17):
                self.grid[y, 11] = 1
    
    def draw(self, surface):
        # Draw background
        surface.fill(WHITE)
        pygame.draw.rect(surface, (240, 240, 240),
                        (CELL_SIZE, CELL_SIZE,
                         18 * CELL_SIZE, 18 * CELL_SIZE))
        
        # Draw grid lines
        for x in range(1, 19):
            for y in range(1, 19):
                rect = pygame.Rect(
                    x * CELL_SIZE, y * CELL_SIZE,
                    CELL_SIZE, CELL_SIZE
                )
                pygame.draw.rect(surface, (245, 245, 245), rect, 1)
        
        # Draw rooms
        for room in self.rooms.values():
            room.draw(surface)
        
        # Draw docking station
        self.draw_docking_station(surface)
        
        # Draw walls
        self.draw_walls(surface)
        
        # Draw obstacles (after rooms and walls, before robot and path)
        self.obstacle_manager.draw(surface)
    
    def draw_docking_station(self, surface):
        x, y = self.docking_station
        dock_rect = pygame.Rect(
            x * CELL_SIZE - 5,
            y * CELL_SIZE - 5,
            CELL_SIZE + 10,
            CELL_SIZE + 10
        )
        pygame.draw.rect(surface, (50, 50, 50), dock_rect, border_radius=5)
        pygame.draw.rect(surface, (70, 70, 70),
                        (x * CELL_SIZE, y * CELL_SIZE - 2,
                         CELL_SIZE, CELL_SIZE + 4),
                        border_radius=3)
        
        # Draw charging contacts
        contact_y = y * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.rect(surface, (220, 220, 0),
                        (x * CELL_SIZE + 8, contact_y - 3, 4, 6))
        pygame.draw.rect(surface, (220, 220, 0),
                        (x * CELL_SIZE + CELL_SIZE - 12, contact_y - 3, 4, 6))
        
        # Power indicator
        power_light_pos = (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + 8)
        pygame.draw.circle(surface, (0, 255, 0), power_light_pos, 3)
    
    def draw_walls(self, surface):
        # Draw all walls based on grid
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if self.grid[y, x] == 1:
                    pygame.draw.rect(surface, BLACK,
                                   (x * CELL_SIZE, y * CELL_SIZE,
                                    CELL_SIZE, CELL_SIZE), 2)
    
    def get_room_at(self, x, y):
        """Return the room ID that contains the given position, or None if not in a room"""
        for room_id, room in self.rooms.items():
            if room.contains_point(x, y):
                return room_id
        return None

class UI:
    def __init__(self, house):
        self.house = house
        self.buttons = {}
    
    def draw(self, surface, robot):
        # Draw UI background
        pygame.draw.rect(surface, UI_BG, 
                        (0, WINDOW_HEIGHT - UI_HEIGHT, WINDOW_WIDTH, UI_HEIGHT))
        
        # Draw gradient
        gradient = pygame.Surface((WINDOW_WIDTH, 5), pygame.SRCALPHA)
        gradient.fill((0, 0, 0, 50))
        surface.blit(gradient, (0, WINDOW_HEIGHT - UI_HEIGHT - 5))
        
        self.draw_dock_status(surface, robot)
        self.draw_room_buttons(surface, robot)
        self.draw_dock_button(surface)
        self.draw_status_bar(surface, robot)
        
        return self.buttons
    
    def draw_dock_status(self, surface, robot):
        font = pygame.font.SysFont("Arial", 22, bold=True)
        dock_status_rect = pygame.Rect(20, WINDOW_HEIGHT - UI_HEIGHT + 15, 140, 45)
        pygame.draw.rect(surface, (30, 30, 35), dock_status_rect, border_radius=6)
        pygame.draw.rect(surface, (50, 50, 55), dock_status_rect, 2, border_radius=6)
        
        status = "DOCKED" if robot.docked else "UNDOCKED"
        color = GREEN if robot.docked else (YELLOW if robot.currently_docking else RED)
        
        label = font.render(status, True, color)
        rect = label.get_rect(center=dock_status_rect.center)
        surface.blit(label, rect)
    
    def draw_room_buttons(self, surface, robot):
        """Draw the room selection buttons"""
        button_width = 50
        button_height = 45
        button_margin = 8
        button_y = WINDOW_HEIGHT - UI_HEIGHT + 15
        
        # Draw "Add to Queue" label
        font = pygame.font.SysFont("Arial", 20, bold=True)
        label = font.render("Add to Queue:", True, WHITE)
        surface.blit(label, (180, button_y - 22))
        
        # Draw room buttons
        x_pos = 180
        for room_id, room in self.house.rooms.items():
            self.buttons[room_id] = pygame.Rect(x_pos, button_y, button_width, button_height)
            
            # Draw button shadow
            pygame.draw.rect(surface, DARK_GRAY, 
                           (x_pos+2, button_y+2, button_width, button_height), 
                           border_radius=8)
            
            # Check button state
            in_queue = room_id in robot.task_queue
            is_target = room_id == robot.target_room
            
            button_color = room.color
            if is_target:
                # Flashing effect for current target
                flash = (pygame.time.get_ticks() // 500) % 2
                if flash:
                    button_color = tuple(min(255, c + 50) for c in button_color)
                
                # Highlight border
                pygame.draw.rect(surface, WHITE,
                               (x_pos-2, button_y-2, button_width+4, button_height+4),
                               2, border_radius=10)
            
            # Draw button
            pygame.draw.rect(surface, button_color, self.buttons[room_id], border_radius=8)
            pygame.draw.rect(surface, BLACK, self.buttons[room_id], 1, border_radius=8)
            
            # Draw room label
            font = pygame.font.SysFont("Arial", 22, bold=True)
            label = font.render(room_id, True, BLACK)
            label_rect = label.get_rect(center=(x_pos + button_width//2, button_y + button_height//2))
            surface.blit(label, label_rect)
            
            # Draw queue indicator
            if in_queue:
                queue_pos = robot.task_queue.index(room_id) + 1
                small_font = pygame.font.SysFont("Arial", 16)
                
                # Add indicator background
                indicator_bg = pygame.Rect(x_pos + 5, button_y + button_height - 20, 20, 18)
                pygame.draw.rect(surface, (50, 50, 50, 180), indicator_bg, border_radius=4)
                
                # Add queue number
                queue_label = small_font.render(f"#{queue_pos}", True, WHITE)
                queue_rect = queue_label.get_rect(center=(x_pos + 15, button_y + button_height - 11))
                surface.blit(queue_label, queue_rect)
            
            x_pos += button_width + button_margin
    
    def draw_dock_button(self, surface):
        """Draw the return to dock button"""
        button_width = 180
        button_y = WINDOW_HEIGHT - UI_HEIGHT + 15
        x_pos = WINDOW_WIDTH - button_width - 20
        
        self.buttons['dock'] = pygame.Rect(x_pos, button_y, button_width, 45)
        
        # Draw button shadow and body
        pygame.draw.rect(surface, DARK_GRAY,
                        (x_pos+2, button_y+2, button_width, 45),
                        border_radius=8)
        pygame.draw.rect(surface, (100, 200, 100),
                        self.buttons['dock'], border_radius=8)
        pygame.draw.rect(surface, BLACK,
                        self.buttons['dock'], 1, border_radius=8)
        
        # Draw label
        font = pygame.font.SysFont("Arial", 22, bold=True)
        label = font.render("RETURN TO DOCK", True, BLACK)
        label_rect = label.get_rect(center=self.buttons['dock'].center)
        surface.blit(label, label_rect)
    
    def draw_status_bar(self, surface, robot):
        """Draw the status bar at the bottom"""
        if robot.target_room or robot.currently_docking or robot.task_queue:
            status_font = pygame.font.SysFont("Arial", 22, bold=True)
            
            # Create status text
            if robot.currently_docking:
                status_text = "Returning to dock"
            elif robot.target_room:
                status_text = f"Vacuuming: {robot.target_room}"
                if robot.vacuum_complete:
                    status_text = f"Cleaned: {robot.target_room}"
            else:
                status_text = "Idle"
            
            # Draw status bar
            status_bar_rect = pygame.Rect(0, WINDOW_HEIGHT - 30, WINDOW_WIDTH, 30)
            pygame.draw.rect(surface, (0, 0, 150), status_bar_rect)
            
            # Draw status text
            label = status_font.render(status_text, True, WHITE)
            surface.blit(label, (20, WINDOW_HEIGHT - 26))

class DetectionLogger:
    def __init__(self, log_file='detections.txt'):
        self.log_file = log_file
        self.detected_objects = {}  # { room_id: set(object_names) }

    def log_object(self, room_id, object_name):
        """Log detected object if not already logged."""
        if room_id not in self.detected_objects:
            self.detected_objects[room_id] = set()

        if object_name not in self.detected_objects[room_id]:
            self.detected_objects[room_id].add(object_name)
            self.save_log()

    def save_log(self):
        """Save all detections into one neat file."""
        with open(self.log_file, 'w') as file:
            for room_id in sorted(self.detected_objects.keys()):
                file.write(f"Room {room_id}:\n")
                for obj_name in sorted(self.detected_objects[room_id]):
                    file.write(f"- {obj_name}\n")
                file.write("\n")  # Add a blank line between rooms

class Robot:
    def __init__(self, x, y, house):
        self.x = x
        self.y = y
        self.house = house
        self.path = []
        self.current_path_index = 0
        self.move_timer = 0
        self.move_delay = 150
        self.target_room = None
        self.vacuum_mode = False
        self.vacuum_complete = False
        self.cleaned_cells = set()
        self.task_queue = []
        self.currently_docking = False
        self.docked = True
        self.object_detector = ObjectDetector()
        self.detection_done = set()
        self.detection_logger = DetectionLogger()


    def add_to_queue(self, room_id):
        """Add a room to the cleaning queue"""
        if room_id not in self.task_queue and room_id in self.house.rooms:
            self.task_queue.append(room_id)
            # If robot is docked and idle, start the first task
            if self.docked and not self.target_room and not self.path:
                self.start_next_task()
    
    def start_next_task(self):
        """Start the next task in queue"""
        if not self.task_queue:
            # If no tasks left, return to dock if not already docked
            if not self.docked and not self.currently_docking:
                self.return_to_dock()
            return
        
        # Undock first if needed
        if self.docked:
            self.docked = False
        
        # Get the next room to clean
        next_room = self.task_queue[0]
        self.task_queue.pop(0)  # Remove from queue
        
        # Set this room as the target
        self.target_room = next_room
        self.vacuum_mode = True
        self.vacuum_complete = False
        self.cleaned_cells = set()
        
        # Generate vacuum path
        self.path = self.generate_vacuum_path(next_room)
        self.current_path_index = 0
    
    def return_to_dock(self):
        """Return to the docking station"""
        print("Returning to dock...")
        self.currently_docking = True
        self.vacuum_mode = False
        self.target_room = None
        
        # Find path to docking station
        dock_path = self.astar((self.x, self.y), self.house.docking_station)
        if dock_path:
            print(f"Found path to dock: {dock_path}")
            self.path = dock_path
            self.current_path_index = 0
        else:
            print("Cannot find path to docking station!")
            self.currently_docking = False
    def load_best_image(self, obstacle_type):
        try:
            # Expected filename format: baseball_cap.png, dust_pan.png, etc.
            filename = f"{obstacle_type.value}.png"
            filepath = os.path.join(BEST_DIR, filename)
            
            if not os.path.exists(filepath):
                print(f"[Error] Best image not found: {filepath}")
                return None

            pil_img = Image.open(filepath).convert('RGB')
            pil_img = pil_img.resize((self.object_detector.img_width, self.object_detector.img_height))
            img_array = np.array(pil_img).astype(np.float32)

            return img_array  # Return as numpy array (H, W, C)

        except Exception as e:
            print(f"[Error] load_best_image failed: {e}")
            return None
        
    def load_random_best_image(self, obstacle_type):
        try:
            folder_name = obstacle_type.value  # e.g., 'baseball_cap'
            folder_path = os.path.join(BEST_DIR, folder_name)

            if not os.path.exists(folder_path):
                print(f"[Error] Best image folder not found: {folder_path}")
                return None

            # List all images in the class folder
            image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

            if not image_files:
                print(f"[Error] No images found in {folder_path}")
                return None

            # Randomly pick one image
            selected_image = random.choice(image_files)
            filepath = os.path.join(folder_path, selected_image)

            # Load and process
            pil_img = Image.open(filepath).convert('RGB')
            pil_img = pil_img.resize((self.object_detector.img_width, self.object_detector.img_height))
            img_array = np.array(pil_img).astype(np.float32)

            return img_array

        except Exception as e:
            print(f"[Error] load_random_best_image failed: {e}")
            return None
    def update(self, current_time):
        """Update robot position and check for collisions"""
        if not self.path or self.current_path_index >= len(self.path) - 1:
            if self.vacuum_mode and not self.vacuum_complete and self.path:
                self.vacuum_complete = True
                if self.task_queue:
                    self.start_next_task()
                else:
                    self.return_to_dock()
            elif self.currently_docking:
                # Check if we've reached the dock
                if (self.x, self.y) == self.house.docking_station:
                    print("Reached docking station")
                    self.docked = True
                    self.currently_docking = False
                    self.path = []
                    if self.task_queue:
                        self.start_next_task()
            return
        
        # Move at regular intervals
        if current_time - self.move_timer > self.move_delay:
            self.move_timer = current_time
            
            # Get next position
            next_x, next_y = self.path[self.current_path_index + 1]
            
            # Check if next position is blocked by an obstacle
            if self.house.obstacle_manager.obstacle_grid[next_y][next_x] == 0:
                self.current_path_index += 1
                self.x, self.y = next_x, next_y
                
                # Add current position to cleaned cells if in vacuum mode
                if self.vacuum_mode:
                    self.cleaned_cells.add((self.x, self.y))
            else:
                # Regenerate path if blocked
                if self.vacuum_mode:
                    self.path = self.generate_vacuum_path(self.target_room)
                elif self.currently_docking:
                    self.path = self.astar((self.x, self.y), self.house.docking_station)
                self.current_path_index = 0
        self.scan_surroundings()
        # if self.target_room and self.vacuum_mode and self.target_room not in self.detection_done:
        #     self.perform_room_detection()
        #     self.detection_done.add(self.target_room)

    def get_expected_object_type(self, grid_x, grid_y):
        """Return the expected object type at a given grid position, if any."""
        for room_id, positions in self.house.obstacle_manager.object_positions.items():
            if (grid_x, grid_y) in positions:
                return positions[(grid_x, grid_y)]
        return None

    def scan_surroundings(self):
        """Scan all cells within 1 block radius and detect obstacles."""
        directions = [(-1, -1), (0, -1), (1, -1),
                    (-1,  0), (0,  0), (1,  0),
                    (-1,  1), (0,  1), (1,  1)]

        for dx, dy in directions:
            check_x = self.x + dx
            check_y = self.y + dy

            if 0 <= check_x < GRID_SIZE and 0 <= check_y < GRID_SIZE:
                if self.house.obstacle_manager.obstacle_grid[check_y][check_x] == 1:
                    expected_type = self.get_expected_object_type(check_x, check_y)

                    if expected_type:
                        best_image = self.load_random_best_image(expected_type)

                        if best_image is None:
                            continue

                        predicted_label, confidence = self.object_detector.detect(best_image)

                        # Find the room where this obstacle is
                        room_id = self.house.get_room_at(check_x, check_y)
                        if not room_id:
                            room_id = 'Unknown'

                        # Only log if correctly detected
                        if predicted_label == expected_type.value:
                            print(f"[Success] Detected {predicted_label} in Room {room_id}")
                            self.detection_logger.log_object(room_id, predicted_label)

    def perform_room_detection(self):
        room_id = self.target_room
        print(f"[Detection] Scanning Room {room_id}...")

        room_objects = self.house.obstacle_manager.object_positions.get(room_id, {})

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
                print(f"[Success] {expected_label} correctly detected with {confidence:.2f}% confidence.")
            else:
                print(f"[Mismatch] Expected {expected_label}, but detected {predicted_label} ({confidence:.2f}%).")


    def capture_cell_view(self, grid_x, grid_y):
        try:
            # Capture a small region around the object
            pixel_x = grid_x * CELL_SIZE
            pixel_y = grid_y * CELL_SIZE

            snapshot_rect = pygame.Rect(pixel_x, pixel_y, CELL_SIZE, CELL_SIZE)
            snapshot_surface = screen.subsurface(snapshot_rect).copy()

            # Convert pygame surface to numpy array
            snapshot_array = pygame.surfarray.array3d(snapshot_surface)
            snapshot_array = np.transpose(snapshot_array, (1, 0, 2))  # From (width, height, channels) to (height, width, channels)
            snapshot_array = snapshot_array.astype(np.float32)
            return snapshot_array

        except Exception as e:
            print(f"[Error] capture_cell_view failed: {e}")
            return None

    def draw(self, surface):
        """Draw the robot on the surface"""
        # Check if robot is at docking station
        is_at_dock = (self.x, self.y) == self.house.docking_station
        
        # Convert grid coordinates to pixel coordinates
        center_x = int(self.x * CELL_SIZE + CELL_SIZE // 2)
        center_y = int(self.y * CELL_SIZE + CELL_SIZE // 2)
        
        # Draw robot body (vacuum cleaner)
        robot_radius = CELL_SIZE // 2 - 3
        
        # Change color based on vacuum mode and docking status
        body_color = RED
        if self.docked or is_at_dock:
            body_color = (50, 180, 50)  # Green when docked
        elif self.vacuum_mode:
            if self.vacuum_complete:
                body_color = GREEN  # Green when cleaning is complete
            else:
                body_color = BLUE  # Blue when actively cleaning
        elif self.currently_docking:
            body_color = YELLOW  # Yellow when returning to dock
        
        # Draw vacuum cleaner body
        pygame.draw.circle(surface, body_color, (center_x, center_y), robot_radius)
        pygame.draw.circle(surface, (200, 200, 200), (center_x, center_y), robot_radius - 4)
        
        # Draw vacuum cleaner details
        if self.vacuum_mode:
            # Draw vacuum base
            pygame.draw.ellipse(surface, DARK_GRAY, 
                              (center_x - 10, center_y + 2, 20, 8))
            
            # Draw brushes (only when actively cleaning)
            if not self.vacuum_complete:
                # Rotating brushes based on time
                brush_angle = pygame.time.get_ticks() / 50
                for i in range(4):
                    angle = brush_angle + i * (math.pi / 2)
                    end_x = center_x + 8 * math.cos(angle)
                    end_y = center_y + 8 * math.sin(angle)
                    pygame.draw.line(surface, BLACK, (center_x, center_y), (end_x, end_y), 2)
        
        # Draw robot eyes
        pygame.draw.circle(surface, BLACK, (center_x - 6, center_y - 5), 3)
        pygame.draw.circle(surface, BLACK, (center_x + 6, center_y - 5), 3)
        
        # Draw direction indicator
        pygame.draw.line(surface, BLACK, (center_x, center_y), 
                         (center_x, center_y - robot_radius), 3)
    
    def set_target_room(self, room_id):
        """This method is kept for backwards compatibility but uses add_to_queue instead"""
        self.add_to_queue(room_id)
        
    def draw_path(self, surface):
        """Draw the path that the robot will follow"""
        if not self.path or len(self.path) <= 1:
            return
        
        # Draw a semi-transparent path
        path_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT - UI_HEIGHT), pygame.SRCALPHA)
        
        # Draw cleaned cells with a light color
        if self.vacuum_mode:
            for cell in self.cleaned_cells:
                cell_x, cell_y = cell
                cell_rect = pygame.Rect(
                    cell_x * CELL_SIZE + 2, 
                    cell_y * CELL_SIZE + 2,
                    CELL_SIZE - 4, 
                    CELL_SIZE - 4
                )
                # Use a more visible color for cleaned cells
                pygame.draw.rect(path_surface, (100, 255, 100, 80), cell_rect)
                
                # Add a subtle "cleaned" pattern
                for i in range(3):
                    line_start = (cell_x * CELL_SIZE + 10, 
                                 cell_y * CELL_SIZE + 10 + i * 5)
                    line_end = (cell_x * CELL_SIZE + CELL_SIZE - 10,
                               cell_y * CELL_SIZE + 10 + i * 5)
                    pygame.draw.line(path_surface, (255, 255, 255, 100), 
                                    line_start, line_end, 1)
        
        # Draw the vacuum path first (behind the main path line)
        if self.vacuum_mode and not self.vacuum_complete:
            # Draw a faint preview of the entire path
            for i in range(self.current_path_index, len(self.path) - 1):
                start_x, start_y = self.path[i]
                end_x, end_y = self.path[i + 1]
                
                start_pos = (start_x * CELL_SIZE + CELL_SIZE // 2, 
                            start_y * CELL_SIZE + CELL_SIZE // 2)
                end_pos = (end_x * CELL_SIZE + CELL_SIZE // 2, 
                          end_y * CELL_SIZE + CELL_SIZE // 2)
                
                # Very faint blue line for future path
                pygame.draw.line(path_surface, (0, 100, 255, 50), start_pos, end_pos, 1)
            
            # Draw dots for each point in the path
            for i in range(self.current_path_index, len(self.path)):
                x, y = self.path[i]
                point_pos = (x * CELL_SIZE + CELL_SIZE // 2, 
                           y * CELL_SIZE + CELL_SIZE // 2)
                pygame.draw.circle(path_surface, (0, 0, 255, 40), point_pos, 2)
        
        # Draw the actual path (more prominent)
        # Only draw a limited portion of the path for better visibility
        display_limit = 50  # Maximum number of path segments to display
        start_idx = max(0, self.current_path_index - 5)  # Show a bit of history
        end_idx = min(len(self.path) - 1, start_idx + display_limit)
        
        for i in range(start_idx, end_idx):
            if i + 1 >= len(self.path):
                break
                
            start_x, start_y = self.path[i]
            end_x, end_y = self.path[i + 1]
            
            start_pos = (start_x * CELL_SIZE + CELL_SIZE // 2, 
                        start_y * CELL_SIZE + CELL_SIZE // 2)
            end_pos = (end_x * CELL_SIZE + CELL_SIZE // 2, 
                      end_y * CELL_SIZE + CELL_SIZE // 2)
            
            # Use different colors for vacuum path
            line_color = (0, 0, 255, 150)  # Default blue
            if self.vacuum_mode:
                if i < self.current_path_index:
                    line_color = (0, 200, 0, 130)  # Green for completed path
                else:
                    line_color = (0, 100, 255, 180)  # Lighter blue for vacuum path
            
            line_width = 2
            if i == self.current_path_index:
                line_width = 3  # Make current segment thicker
                
            pygame.draw.line(path_surface, line_color, start_pos, end_pos, line_width)
            
            # Add little circles at each path point
            if i >= self.current_path_index - 1:
                circle_color = line_color
                pygame.draw.circle(path_surface, circle_color, start_pos, 3)
        
        # Draw the current position point
        if self.current_path_index < len(self.path):
            current_point = self.path[self.current_path_index]
            current_pos = (current_point[0] * CELL_SIZE + CELL_SIZE // 2, 
                          current_point[1] * CELL_SIZE + CELL_SIZE // 2)
            pygame.draw.circle(path_surface, (255, 100, 100, 200), current_pos, 5)
        
        # Draw the last point (destination)
        if len(self.path) > 0:
            last_point = self.path[-1]
            dest_point = (last_point[0] * CELL_SIZE + CELL_SIZE // 2, 
                         last_point[1] * CELL_SIZE + CELL_SIZE // 2)
            
            dest_color = (0, 255, 0, 200)
            if self.vacuum_mode and not self.vacuum_complete:
                dest_color = (255, 165, 0, 200)  # Orange for vacuum destination
            
            pygame.draw.circle(path_surface, dest_color, dest_point, 6)
        
        surface.blit(path_surface, (0, 0))

    def heuristic(self, a, b):
        """Calculate Manhattan distance heuristic (no diagonals)"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_neighbors(self, node):
        """Get valid neighboring cells (no diagonals)"""
        x, y = node
        neighbors = []
        
        # Check only cardinal directions (no diagonals)
        directions = [
            (0, 1),   # Up
            (1, 0),   # Right
            (0, -1),  # Down
            (-1, 0)   # Left
        ]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # Check if position is within grid bounds and not a wall or obstacle
            if (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and 
                self.house.grid[ny][nx] == 0 and
                self.house.obstacle_manager.obstacle_grid[ny][nx] == 0):
                neighbors.append((nx, ny))
        
        return neighbors
    
    def astar(self, start, goal):
        """Improved A* pathfinding algorithm implementation"""
        open_set = PriorityQueue()
        open_set.put((0, 0, start))  # (f_score, counter, position)
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        open_set_hash = {start}
        counter = 1  # Used as a tiebreaker for equal f_scores
        
        # Track visited nodes for visualization
        visited = set()
        
        while not open_set.empty():
            _, _, current = open_set.get()
            open_set_hash.remove(current)
            visited.add(current)
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            for neighbor in self.get_neighbors(current):
                # Cost is always 1 for cardinal moves
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    
                    if neighbor not in open_set_hash:
                        counter += 1
                        open_set.put((f_score[neighbor], counter, neighbor))
                        open_set_hash.add(neighbor)
        
        # No path found
        return []
    
    def generate_vacuum_path(self, room_id):
        """Generate an efficient path that covers the entire room for vacuum cleaning"""
        if room_id not in self.house.rooms:
            return []
        
        # Get room coordinates
        room = self.house.rooms[room_id]
        room_x, room_y, width, height = room.get_coords()
        
        print(f"Room {room_id} coordinates: x={room_x}, y={room_y}, width={width}, height={height}")
        
        # Find all accessible cells in the room including edges
        room_cells = []
        
        # Scan the room area, including one extra cell in each direction to catch borders
        for y in range(room_y-1, room_y + height+1):
            for x in range(room_x-1, room_x + width+1):
                # Only add cells that are walkable (not walls)
                if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and self.house.grid[y][x] == 0:
                    # Check if this cell is actually in the room (for doorways/entry points)
                    is_in_room = (
                        (room_x <= x < room_x + width and room_y <= y < room_y + height) or
                        # Special cases for doorways
                        (room_id == 'A' and 2 <= x <= 3 and y == 10) or  # A's entrance from B
                        (room_id == 'B' and 2 <= x <= 3 and y == 9) or   # B's entrance from A
                        (room_id == 'C' and x == 11 and 8 <= y <= 9) or  # C's entrance from B
                        (room_id == 'D' and 16 <= x <= 17 and y == 10)   # D's entrance from C
                    )
                    
                    if is_in_room:
                        room_cells.append((x, y))
        
        if not room_cells:
            print(f"No accessible cells found in room {room_id}!")
            return []
        
        # Check if robot is already in the room
        robot_pos = (self.x, self.y)
        robot_in_room = False
        for cell in room_cells:
            if cell == robot_pos:
                robot_in_room = True
                break
        
        # Find path to reach the room if robot is not already there
        path_to_room = []
        if not robot_in_room:
            # Define entry points for each room
            entry_points = []
            
            # Add potential entry points for each room
            if room_id == 'A':
                entry_points = [(2, 10), (3, 10)]  # Stairs from B to A
            elif room_id == 'B':
                entry_points = [(2, 9), (3, 9)]    # Stairs from A to B
            elif room_id == 'C':
                entry_points = [(11, 8), (11, 9)]  # Door between B and C
            elif room_id == 'D':
                entry_points = [(16, 10), (17, 10)] # Stairs from C to D
            
            # If entry points were specified, find the best one
            if entry_points:
                best_path = []
                for point in entry_points:
                    path = self.astar(robot_pos, point)
                    if path and (not best_path or len(path) < len(best_path)):
                        best_path = path
                
                if best_path:
                    path_to_room = best_path
            
            # If no valid entry point found, try to pathfind to any cell in the room
            if not path_to_room:
                # Sort room cells by distance from robot
                sorted_room_cells = sorted(room_cells, 
                                         key=lambda c: abs(c[0] - self.x) + abs(c[1] - self.y))
                
                # Try to find a path to any cell in the room
                for cell in sorted_room_cells:
                    path = self.astar(robot_pos, cell)
                    if path:
                        path_to_room = path
                        break
            
            if not path_to_room:
                print(f"Cannot find path to room {room_id}!")
                return []  # Can't reach the room
        
        # Starting point for the vacuum path within the room
        start_in_room = path_to_room[-1] if path_to_room else robot_pos
        
        # Sort room cells by row then column for efficient coverage
        row_sorted_cells = sorted(room_cells, key=lambda c: (c[1], c[0]))
        
        # Group cells by row
        rows = {}
        for cell in row_sorted_cells:
            x, y = cell
            if y not in rows:
                rows[y] = []
            rows[y].append(cell)
        
        # Find the row of our starting point
        start_row = start_in_room[1]
        
        # Sort rows by distance from starting row
        sorted_rows = sorted(rows.keys())
        
        # Find starting row index
        start_row_idx = -1
        for i, row in enumerate(sorted_rows):
            if row == start_row:
                start_row_idx = i
                break
        
        if start_row_idx == -1:
            # Default to middle if start row not found
            start_row_idx = len(sorted_rows) // 2
        
        # Reorder rows to start from the robot's row and go up and down
        reordered_rows = []
        
        # Add current row first
        if start_row_idx >= 0 and start_row_idx < len(sorted_rows):
            reordered_rows.append(sorted_rows[start_row_idx])
        
        # Add rows above and below alternately
        max_offset = max(start_row_idx, len(sorted_rows) - start_row_idx - 1)
        for offset in range(1, max_offset + 1):
            # Add row below if possible
            if start_row_idx + offset < len(sorted_rows):
                reordered_rows.append(sorted_rows[start_row_idx + offset])
            
            # Add row above if possible
            if start_row_idx - offset >= 0:
                reordered_rows.append(sorted_rows[start_row_idx - offset])
        
        # Create coverage path with improved pattern
        coverage_path = []
        
        # Process rows in optimized order
        for i, row in enumerate(reordered_rows):
            row_cells = rows[row]
            
            # For the starting row, determine direction based on robot position
            if i == 0:  # Starting row
                if start_in_room[0] < sum(x for x, _ in row_cells) / len(row_cells):
                    row_cells.sort(key=lambda c: c[0])  # Left to right
                else:
                    row_cells.sort(key=lambda c: -c[0])  # Right to left
            else:
                # For other rows, alternate direction
                if i % 2 == 0:
                    row_cells.sort(key=lambda c: c[0])  # Left to right
                else:
                    row_cells.sort(key=lambda c: -c[0])  # Right to left
            
            coverage_path.extend(row_cells)
        
        # Now create a continuous path through all these points using A*
        continuous_path = [start_in_room]
        visited = {start_in_room}
        
        # Remove starting point from coverage path if it's there
        if start_in_room in coverage_path:
            coverage_path.remove(start_in_room)
        
        # Process cells in the order determined by our coverage pattern
        current_pos = start_in_room
        while coverage_path:
            # Find the closest unvisited cell in the coverage path
            closest_cell = min(coverage_path, 
                             key=lambda c: abs(c[0] - current_pos[0]) + abs(c[1] - current_pos[1]))
            
            # Find path from current position to closest cell
            sub_path = self.astar(current_pos, closest_cell)
            if sub_path and len(sub_path) > 1:  # If path exists and has more than just start
                # Add all cells in the path except the first one (current position)
                continuous_path.extend(sub_path[1:])
                # Update current position to the last cell in the path
                current_pos = sub_path[-1]
                # Remove the target cell from coverage path
                coverage_path.remove(closest_cell)
            else:
                # If no path found, try the next closest cell
                coverage_path.remove(closest_cell)
        
        # Combine path to room and continuous path within room
        final_path = path_to_room + continuous_path[1:] if path_to_room else continuous_path
        print(f"Generated path with {len(final_path)} points for room {room_id}")
        
        return final_path

def draw_environment():
    """Draw the house layout"""
    # Draw background
    screen.fill(WHITE)
    
    # Draw the house base - light gray background
    pygame.draw.rect(screen, (240, 240, 240), 
                   (1 * CELL_SIZE, 1 * CELL_SIZE,
                    18 * CELL_SIZE, 18 * CELL_SIZE))
    
    # Draw grid lines (for better visualization of cells)
    # This will help show which cells are being cleaned
    for x in range(1, 19):
        for y in range(1, 19):
            rect = pygame.Rect(
                x * CELL_SIZE, y * CELL_SIZE,
                CELL_SIZE, CELL_SIZE
            )
            pygame.draw.rect(screen, (245, 245, 245), rect, 1)
    
    # Draw all rooms
    for room_id, room_data in ROOMS.items():
        x, y, width, height = room_data['coords']
        # Draw room with slight expansion to avoid white gaps
        pygame.draw.rect(screen, room_data['color'], 
                       (x * CELL_SIZE - 1, y * CELL_SIZE - 1, 
                        width * CELL_SIZE + 2, height * CELL_SIZE + 2))
    
    # Draw stairs (special passages)
    # Stairs from B to A
    pygame.draw.rect(screen, (220, 220, 220), 
                   (2 * CELL_SIZE, 9 * CELL_SIZE, 
                    2 * CELL_SIZE, 2 * CELL_SIZE))
    
    # Stairs from C to D
    pygame.draw.rect(screen, (220, 220, 220), 
                   (16 * CELL_SIZE, 9 * CELL_SIZE, 
                    2 * CELL_SIZE, 2 * CELL_SIZE))
    
    # Draw docking station
    dock_x, dock_y = DOCKING_STATION
    dock_rect = pygame.Rect(
        dock_x * CELL_SIZE - 5, 
        dock_y * CELL_SIZE - 5,
        CELL_SIZE + 10, 
        CELL_SIZE + 10
    )
    
    # Draw the docking station base
    pygame.draw.rect(screen, (50, 50, 50), dock_rect, border_radius=5)
    pygame.draw.rect(screen, (70, 70, 70), 
                   (dock_x * CELL_SIZE, dock_y * CELL_SIZE - 2,
                    CELL_SIZE, CELL_SIZE + 4), 
                   border_radius=3)
    
    # Draw charging contacts
    contact_y = dock_y * CELL_SIZE + CELL_SIZE // 2
    pygame.draw.rect(screen, (220, 220, 0), 
                   (dock_x * CELL_SIZE + 8, contact_y - 3, 4, 6))
    pygame.draw.rect(screen, (220, 220, 0), 
                   (dock_x * CELL_SIZE + CELL_SIZE - 12, contact_y - 3, 4, 6))
    
    # Add a power indicator light
    power_light_pos = (dock_x * CELL_SIZE + CELL_SIZE // 2, dock_y * CELL_SIZE + 8)
    pygame.draw.circle(screen, (0, 255, 0), power_light_pos, 3)
    
    # Draw walls
    for wall in WALL_LINES:
        x1, y1, x2, y2 = wall
        start_pos = (x1 * CELL_SIZE, y1 * CELL_SIZE)
        end_pos = (x2 * CELL_SIZE, y2 * CELL_SIZE)
        pygame.draw.line(screen, BLACK, start_pos, end_pos, 5)
    
    # Make sure wall corners are properly filled
    for wall1 in WALL_LINES:
        for wall2 in WALL_LINES:
            x1, y1, x2, y2 = wall1
            x3, y3, x4, y4 = wall2
            
            # Check for vertical-horizontal intersections
            if (x1 == x2 and y3 == y4 and 
                min(x3, x4) <= x1 <= max(x3, x4) and
                min(y1, y2) <= y3 <= max(y1, y2)):
                pygame.draw.rect(screen, BLACK, 
                               (x1 * CELL_SIZE - 2, y3 * CELL_SIZE - 2, 5, 5))
    
    # Draw doors
    for door in DOORS:
        x, y, width, height = door
        
        # Draw door
        door_rect = pygame.Rect(
            x * CELL_SIZE, y * CELL_SIZE,
            width * CELL_SIZE, height * CELL_SIZE
        )
        
        # Get the color of the adjacent room for the door
        door_color = WHITE
        for r_id, r_data in ROOMS.items():
            rx, ry, rw, rh = r_data['coords']
            # Check if door is adjacent to this room
            if ((x + width == rx or x == rx + rw) and 
                (y >= ry and y < ry + rh)):
                door_color = r_data['color']
                break
            if ((y + height == ry or y == ry + rh) and 
                (x >= rx and x < rx + rw)):
                door_color = r_data['color']
                break
        
        pygame.draw.rect(screen, door_color, door_rect)
        
        # Draw door swing
        if width > height:  # Horizontal door
            center_x = door_rect.centerx
            center_y = door_rect.top
            radius = door_rect.width // 2
            pygame.draw.arc(screen, DARK_GRAY, 
                          (center_x - radius, center_y - radius, 
                           radius * 2, radius * 2), 
                          math.radians(0), math.radians(90), 2)
        else:  # Vertical door
            center_x = door_rect.left
            center_y = door_rect.centery
            radius = door_rect.height // 2
            pygame.draw.arc(screen, DARK_GRAY, 
                          (center_x - radius, center_y - radius, 
                           radius * 2, radius * 2), 
                          math.radians(90), math.radians(180), 2)

def draw_room_labels():
    """Draw simplified room labels"""
    font = pygame.font.SysFont("Arial", 24, bold=True)
    
    for room_id, room_data in ROOMS.items():
        x, y, width, height = room_data['coords']
        
        # Create simple label (just the room letter)
        label = font.render(room_id, True, BLACK)
        
        # Center label in room
        label_rect = label.get_rect(
            center=(x * CELL_SIZE + (width * CELL_SIZE) // 2,
                   y * CELL_SIZE + (height * CELL_SIZE) // 2)
        )
        
        screen.blit(label, label_rect)

def handle_mouse_click(pos, robot, buttons):
    """Handle mouse click events"""
    for room_id, button in buttons.items():
        if button.collidepoint(pos):
            if room_id == 'dock':
                if not robot.docked and not robot.currently_docking:
                    robot.return_to_dock()
            else:
                robot.add_to_queue(room_id)

def main():
    # Initialize the environment
    initialize_grid()
    
    # Debug print of grid
    print_grid_debug()
    
    # Create robot at docking station
    house = House()
    robot = Robot(DOCKING_STATION[0], DOCKING_STATION[1], house)
    ui = UI(house)
    
    running = True
    while running:
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                handle_mouse_click(event.pos, robot, ui.buttons)
        
        # Update robot
        robot.update(current_time)
        
        # Draw everything
        house.draw(screen)  # Use house's draw method instead of draw_environment
        draw_room_labels()
        
        # Draw the path and robot
        robot.draw_path(screen)
        robot.draw(screen)
        
        # Draw UI
        buttons = ui.draw(screen, robot)
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()

if __name__ == "__main__":
    main()