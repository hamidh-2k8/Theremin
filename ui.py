import cv2
import tracker
import logging
import faulthandler
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer
from PySide6 import QtWidgets as qtw

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("UI")
faulthandler.enable()

class HeaderBar(qtw.QWidget):
    def __init__(self):
        super().__init__()

        layout = qtw.QHBoxLayout()
        layout.addWidget(qtw.QLabel("Theremin Control"))
        
        self.startButton = qtw.QPushButton("Start Tracking")
        self.startButton.pressed.connect(self.startTracking)
        layout.addWidget(self.startButton)
        self.setLayout(layout)

    def startTracking(self):
        tracker.startTracking()
        self.startButton.setText("Stop Tracking")
        self.startButton.pressed.disconnect()
        self.startButton.pressed.connect(self.stopTracking)

    def stopTracking(self):
        tracker.stopTracking()
        self.startButton.setText("Start Tracking")
        self.startButton.pressed.disconnect()
        self.startButton.pressed.connect(self.startTracking)
        
class ConfigPanel(qtw.QWidget):

    def __init__(self):
        super().__init__()

        self.layout = qtw.QGridLayout(self)

        self.towerCamSelector = qtw.QComboBox()
        self.baseCamSelector = qtw.QComboBox()
        self.applyButton = qtw.QPushButton("Apply")
        self.refreshButton = qtw.QPushButton("Refresh")

        self.applyButton.clicked.connect(self.applyChanges)

        self.refreshButton.clicked.connect(self.refreshDeviceList)

        self.layout.addWidget(qtw.QLabel("Tower Camera:"), 0, 0)
        self.layout.addWidget(self.towerCamSelector, 0, 1)
        self.layout.addWidget(qtw.QLabel("Base Camera:"), 0, 2)
        self.layout.addWidget(self.baseCamSelector, 0, 3)
        self.layout.addWidget(self.refreshButton, 1, 2)
        self.layout.addWidget(self.applyButton, 1, 3)

    def refreshDeviceList(self):
        self.deviceList = {}
        for dev in tracker.getCaptureDevices():
            self.deviceList[dev.path] = dev.name
        self.towerCamSelector.clear()
        self.towerCamSelector.addItems([f"{name} ({path})" for path, name in self.deviceList.items()])
        self.baseCamSelector.clear()
        self.baseCamSelector.addItems([f"{name} ({path})" for path, name in self.deviceList.items()])

    def applyChanges(self):

        if self.towerCamSelector.currentIndex() == self.baseCamSelector.currentIndex():
            LOGGER.warning("Same camera selected for tower and base!")
            return
        
        newConfig = {
            "tower_id": list(self.deviceList.keys())[self.towerCamSelector.currentIndex()],
            "base_id": list(self.deviceList.keys())[self.baseCamSelector.currentIndex()],
            "frame_width": 800,
            "frame_height": 600, # TODO: Add config for these
            "fps": 30
        }
        tracker.updateConfig(newConfig)
        
class ResultPreviews(qtw.QWidget):

    def __init__(self):
        super().__init__()
        
        self.towerPreview = qtw.QLabel()
        self.basePreview = qtw.QLabel()
        
        self.refreshImages()

        layout = qtw.QGridLayout()
        layout.addWidget(qtw.QLabel("Tower Camera Preview:"), 0, 0)
        layout.addWidget(self.towerPreview, 1, 0)
        layout.addWidget(qtw.QLabel("Base Camera Preview:"), 0, 1)
        layout.addWidget(self.basePreview, 1, 1)
        self.setLayout(layout)
        
        self.refreshTimer = QTimer(self)
        self.refreshTimer.timeout.connect(self.refreshImages)
        self.refreshTimer.start(1000/30)

    def refreshImages(self):

        towerImage = cv2.resize(tracker.towerResult.getAnnotatedImage(), (400, 400))
        self.towerPreview.setPixmap(QPixmap.fromImage(QImage(towerImage.data, 400, 400, towerImage.shape[2] * 400, QImage.Format_RGB888)))

        baseImage = cv2.resize(tracker.baseResult.getAnnotatedImage(), (400, 400))
        self.basePreview.setPixmap(QPixmap.fromImage(QImage(baseImage.data, 400, 400, baseImage.shape[2] * 400, QImage.Format_RGB888)))

class MainWindow(qtw.QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Theremin Control Panel")

        self.setCentralWidget(qtw.QWidget())
        self.centralWidget().setLayout(qtw.QVBoxLayout())
        self.centralWidget().layout().addWidget(HeaderBar())
        self.centralWidget().layout().addWidget(ResultPreviews())
        self.centralWidget().layout().addWidget(ConfigPanel())
        LOGGER.info("UI initialized")
        self.show()
        
    def closeEvent(self, event):
    
        tracker.stopTracking()
        event.accept()


app = qtw.QApplication([])
window = MainWindow()
app.exec()