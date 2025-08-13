# robot_gui.py
import sys
from PyQt5.QtWidgets import QApplication
from integratedfullupdatedfinal import RobotVoiceCommandSystem  # Assuming your main PyQt5 window is here

def main():
    app = QApplication(sys.argv)
    main_window = RobotVoiceCommandSystem()
    main_window.showFullScreen()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
