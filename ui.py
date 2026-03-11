import cv2
import tracker
import logging
import faulthandler
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer, Qt
from PySide6 import QtWidgets as qtw

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("UI")
faulthandler.enable()

class HeaderBar(qtw.QWidget):
    def __init__(self):
        super().__init__()

        layout = qtw.QHBoxLayout()
        self.title = qtw.QLabel("Theremin Control")
        self.title.setStyleSheet("font-weight: bold; font-size: 36px;")
        layout.addWidget(self.title)

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
            "fps": 60
        }
        tracker.updateConfig(newConfig)

class ResultBargraphs(qtw.QWidget):

    def __init__(self):
        super().__init__()

        layout = qtw.QHBoxLayout()
        self.setLayout(layout)

        self.barList = {
            "lift": qtw.QProgressBar(self),
            "pitch": qtw.QProgressBar(self),
            "yaw": qtw.QProgressBar(self),
            "roll": qtw.QProgressBar(self),
            "indexCurl": qtw.QProgressBar(self),
            "middleCurl": qtw.QProgressBar(self),
            "ringCurl": qtw.QProgressBar(self),
            "pinkyCurl": qtw.QProgressBar(self),
            "fullCurl": qtw.QProgressBar(self),
            "spread": qtw.QProgressBar(self),
            "pinch": qtw.QProgressBar(self)
        }

        for bar in self.barList.values():
            bar.setRange(0, 1)
            bar.setOrientation(Qt.Orientation.Vertical)
            bar.setTextVisible(False)
            layout.addWidget(bar)

    def setBars(self, values: dict):
        print(values)
        for key, value in values.items():
            if key in self.barList:
                self.barList[key].setValue(value)

class ResultPreviews(qtw.QWidget):

    def __init__(self, camType: tracker.CameraType):
        super().__init__()

        self.camType = camType
        self.preview = qtw.QLabel()

        self.preview.setStyleSheet("border: 4px solid white;")


        layout = qtw.QGridLayout()
        layout.addWidget(qtw.QLabel(f"{'Tower' if self.camType == tracker.CameraType.TOWER else 'Base'} Camera Preview"), 0, 0)
        layout.addWidget(self.preview, 1, 0)
        self.setLayout(layout)
        
        self.resBars = ResultBargraphs()
        layout.addWidget(self.resBars, 2, 0)

        self.refreshImages()
        
        self.refreshTimer = QTimer(self)
        self.refreshTimer.timeout.connect(self.refreshImages)
        self.refreshTimer.start(1000/60)

    def refreshImages(self):

        result = tracker.towerResult if self.camType == tracker.CameraType.TOWER else tracker.baseResult

        image = cv2.resize(result.getAnnotatedImage(), (400, 400))
        self.preview.setPixmap(QPixmap.fromImage(QImage(image.data, 400, 400, image.shape[2] * 400, QImage.Format_RGB888)))
        
        self.resBars.setBars(result.processed)

class MainWindow(qtw.QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Theremin Control Panel")

        self.setCentralWidget(qtw.QWidget())
        self.centralWidget().setLayout(qtw.QVBoxLayout())
        self.centralWidget().layout().addWidget(HeaderBar())

        self.previewGrid = qtw.QGridLayout()
        self.previewGrid.addWidget(ResultPreviews(tracker.CameraType.TOWER), 0, 0)
        self.previewGrid.addWidget(ResultPreviews(tracker.CameraType.BASE), 0, 1)
        self.centralWidget().layout().addLayout(self.previewGrid)

        self.centralWidget().layout().addWidget(ConfigPanel())
        LOGGER.info("UI initialized")
        self.show()
        
    def closeEvent(self, event):
    
        tracker.stopTracking()
        event.accept()


app = qtw.QApplication([])
window = MainWindow()
app.exec()