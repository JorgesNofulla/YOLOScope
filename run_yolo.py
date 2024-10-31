import sys
import os
import cv2
import numpy as np
import tempfile
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
    QHBoxLayout, QScrollArea, QGraphicsDropShadowEffect, QStatusBar, QMessageBox, QSizePolicy,
    QStyle, QSlider, QStackedWidget
)
from PyQt5.QtGui import QPixmap, QImage, QIcon, QPalette
from PyQt5.QtCore import Qt, pyqtSignal, QEvent, QUrl, QObject, QThread
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from style import STYLE  # Ensure your STYLE string includes any necessary styles

class YOLOv4App(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.loadModel()
        self.setStyleSheet(STYLE)  # Apply the stylesheet

        # Initialize detection data storage
        self.detections = []
        self.original_image = None
        self.current_highlight_class = None  # To keep track of the selected class
        self.displayed_image = None  # To store the current displayed image
        self.processed_video_path = None  # To store the path of the processed video
        self.processing_video = False  # Flag to indicate if a video is being processed

    def initUI(self):
        self.setWindowTitle('Object Detection')
        self.setWindowIcon(QIcon('assets/icon.png'))  # Set the window icon
        self.setGeometry(100, 100, 1200, 800)  # Adjusted window size for better layout

        main_layout = QVBoxLayout()

        # Title
        title = QLabel("Object Detection")
        title.setObjectName("TitleLabel")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        # Upper layout: Upload Buttons and Legend
        upper_layout = QHBoxLayout()

        # Upload Image Button with Icon
        self.uploadButton = QPushButton(' Upload Image')
        self.uploadButton.setIcon(QIcon('icons/upload.png'))  # Ensure 'icons/upload.png' exists
        self.uploadButton.clicked.connect(self.uploadImage)
        upper_layout.addWidget(self.uploadButton)

        # Upload Video Button with Icon
        self.uploadVideoButton = QPushButton(' Upload Video')
        self.uploadVideoButton.setIcon(QIcon('icons/upload_video.png'))  # Ensure 'icons/upload_video.png' exists
        self.uploadVideoButton.clicked.connect(self.uploadVideo)
        upper_layout.addWidget(self.uploadVideoButton)

        # Spacer
        upper_layout.addStretch()

        # Legend Scroll Area
        self.scrollArea = QScrollArea()
        self.scrollArea.setFixedHeight(200)
        self.scrollArea.setWidgetResizable(True)
        self.legendWidget = QWidget()
        self.legendLayout = QVBoxLayout(self.legendWidget)
        self.legendLayout.setSpacing(5)
        self.scrollArea.setWidget(self.legendWidget)
        upper_layout.addWidget(self.scrollArea)

        main_layout.addLayout(upper_layout)

        # Display Stack: Will hold either image or video
        self.displayStack = QStackedWidget()
        main_layout.addWidget(self.displayStack, 1)

        # Image Display with Shadow inside a Scroll Area
        self.imageScrollArea = QScrollArea()
        self.imageScrollArea.setWidgetResizable(True)
        self.imageLabel = QLabel(self)
        self.imageLabel.setObjectName("ImageLabel")
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.imageLabel.setBackgroundRole(QPalette.Base)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(False)  # Prevent stretching

        # Add shadow effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(0)
        shadow.setColor(Qt.gray)
        self.imageLabel.setGraphicsEffect(shadow)

        self.imageScrollArea.setWidget(self.imageLabel)

        # Video Display
        self.videoWidget = QVideoWidget()
        self.videoWidget.setObjectName("VideoWidget")

        # Media Player
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.mediaPlayer.setVideoOutput(self.videoWidget)

        # Add imageScrollArea and videoWidget to the displayStack
        self.displayStack.addWidget(self.imageScrollArea)
        self.displayStack.addWidget(self.videoWidget)

        # Playback Controls
        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.error.connect(self.handleMediaError)

        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addWidget(self.playButton)
        controls_layout.addWidget(self.positionSlider)

        main_layout.addLayout(controls_layout)

        # Initialize Status Bar
        self.statusBar = QStatusBar()
        self.statusBar.showMessage("Ready")  # Initial message
        main_layout.addWidget(self.statusBar)

        self.setLayout(main_layout)

        # Install event filter on the scroll area's viewport to handle resizing
        self.imageScrollArea.viewport().installEventFilter(self)

    def eventFilter(self, source, event):
        if source == self.imageScrollArea.viewport() and event.type() == QEvent.Resize:
            self.rescaleImage()
            return True
        return super().eventFilter(source, event)

    def rescaleImage(self):
        """
        Rescales the displayed image to fit within the scroll area's viewport
        while maintaining the aspect ratio.
        """
        if self.displayed_image is not None:
            self.displayDetectedImage(highlight_class_id=self.current_highlight_class)

    def loadModel(self):
        try:
            self.statusBar.showMessage("Loading YOLOv4 model...")
            self.yolo_model = cv2.dnn.readNetFromDarknet('model/yolov4.cfg', 'model/yolov4.weights')
            with open('model/coco.names', 'r') as f:
                self.class_labels = f.read().strip().split('\n')
            self.model_layers = self.yolo_model.getLayerNames()
            self.output_layers = [self.model_layers[i - 1] for i in self.yolo_model.getUnconnectedOutLayers()]
            # Generate consistent colors for each class (convert to RGB for legend)
            np.random.seed(42)  # Seed for reproducibility
            self.class_colors = {i: tuple(np.random.randint(0, 255, 3).tolist()) for i in range(len(self.class_labels))}
            self.statusBar.showMessage("Model loaded successfully.", 5000)  # Message for 5 seconds
        except Exception as e:
            QMessageBox.critical(self, "Model Loading Error", f"An error occurred while loading the model:\n{str(e)}")
            self.statusBar.showMessage("Model loading failed.", 5000)  # Message for 5 seconds

    def uploadImage(self):
        self.processing_video = False  # We're processing an image
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            self, 'Select Image', '', 'Images (*.png *.xpm *.jpg *.jpeg *.bmp)', options=options)
        if fileName:
            self.imagePath = fileName
            self.original_image = cv2.imread(self.imagePath)
            if self.original_image is None:
                self.statusBar.showMessage("Failed to load image.", 3000)
                self.imageLabel.setText("Failed to load image.")
                return
            self.displayImage(self.original_image)
            self.performDetection()
            self.statusBar.showMessage(f"Image '{os.path.basename(fileName)}' loaded and detection performed.", 5000)  # Message for 5 seconds
            # Switch to image display
            self.displayStack.setCurrentWidget(self.imageScrollArea)
            self.playButton.setEnabled(False)
        else:
            self.statusBar.showMessage("Image upload canceled.", 3000)  # Message for 3 seconds

    def displayImage(self, img):
        """
        Displays the original image in the imageLabel without scaling.
        """
        if img is None:
            self.imageLabel.setText("No image to display.")
            return
        # Store the original image for future scaling
        self.displayed_image = img.copy()

        # Convert image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img_rgb.shape
        bytesPerLine = 3 * width
        qImg = QImage(img_rgb.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)

        # Scale pixmap to fit within scroll area's viewport, keeping aspect ratio
        viewport_size = self.imageScrollArea.viewport().size()
        scaled_pixmap = pixmap.scaled(viewport_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.imageLabel.setPixmap(scaled_pixmap)
        # Do not call adjustSize() to prevent stretching
        # self.imageLabel.adjustSize()

    def performDetection(self):
        if self.original_image is None:
            self.statusBar.showMessage("No image loaded for detection.", 3000)
            return

        test_img = self.original_image.copy()
        scalefactor = 1.0 / 255.0
        new_size = (416, 416)
        blob = cv2.dnn.blobFromImage(test_img, scalefactor, new_size, swapRB=True, crop=False)
        self.yolo_model.setInput(blob)
        obj_detections_in_layers = self.yolo_model.forward(self.output_layers)

        img_height, img_width = test_img.shape[:2]
        class_ids = []
        confidences = []
        boxes = []

        for layer in obj_detections_in_layers:
            for detection in layer:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * img_width)
                    center_y = int(detection[1] * img_height)
                    width = int(detection[2] * img_width)
                    height = int(detection[3] * img_height)
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)
                    boxes.append([x, y, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Initialize detections and detected_classes
        self.detections = []  # Clear previous detections
        detected_classes = set()

        # Check if indices is not empty and is iterable
        if isinstance(indices, (list, np.ndarray)) and len(indices) > 0:
            # Depending on OpenCV version, indices might need to be flattened differently
            if isinstance(indices, tuple):
                indices = indices[0]  # Extract the first element if it's a tuple
            try:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    class_id = class_ids[i]
                    confidence = confidences[i]
                    color = self.class_colors[class_id]
                    detection = {
                        'class_id': class_id,
                        'box': [x, y, w, h],
                        'confidence': confidence,
                        'color': color
                    }
                    self.detections.append(detection)
                    detected_classes.add(class_id)
            except AttributeError:
                # In case indices cannot be flattened
                self.detections = []
                self.statusBar.showMessage("No objects detected in the image.", 5000)
                QMessageBox.information(self, "No Detections", "No objects were detected in the uploaded image.")
                self.displayDetectedImage()  # Display the original image without detections
                return
        else:
            # No detections found
            self.detections = []
            self.statusBar.showMessage("No objects detected in the image.", 5000)  # Message for 5 seconds
            QMessageBox.information(self, "No Detections", "No objects were detected in the uploaded image.")
            self.displayDetectedImage()  # Display the original image without detections
            return

        # If detections are present, display them
        self.displayDetectedImage()
        self.createLegend(detected_classes)

    def displayDetectedImage(self, highlight_class_id=None):
        """
        Displays the image with bounding boxes. Highlights boxes of the selected class if specified.
        """
        if self.original_image is None:
            self.imageLabel.setText("No image to display.")
            return

        img = self.original_image.copy()
        for detection in self.detections:
            class_id = detection['class_id']
            x, y, w, h = detection['box']
            color = detection['color']
            if not self.processing_video and highlight_class_id is not None and class_id == highlight_class_id:
                # Highlighted box (e.g., thicker border and different color)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 3)  # Yellow color for highlight
                label = f'{self.class_labels[class_id]}: {detection["confidence"]:.2f}'
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 255), 2)
            else:
                # Normal box
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                label = f'{self.class_labels[class_id]}: {detection["confidence"]:.2f}'
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)

        # Store the modified image for future scaling
        self.displayed_image = img.copy()

        # Convert image to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img_rgb.shape
        bytesPerLine = 3 * width
        qImg = QImage(img_rgb.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)

        # Scale pixmap to fit within scroll area's viewport, keeping aspect ratio
        viewport_size = self.imageScrollArea.viewport().size()
        scaled_pixmap = pixmap.scaled(viewport_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.imageLabel.setPixmap(scaled_pixmap)
        # Do not call adjustSize() to prevent stretching
        # self.imageLabel.adjustSize()

    def createLegend(self, class_ids):
        """
        Creates a clickable legend based on detected classes.
        """
        if self.processing_video:
            return  # Skip creating the legend when processing video

        # Clear the existing legend
        while self.legendLayout.count():
            item = self.legendLayout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self.clearLayout(item.layout())

        # Add new legend items
        for class_id in sorted(class_ids):
            color = self.class_colors[class_id]
            # Convert BGR to RGB for correct color display in legend
            r, g, b = color[2], color[1], color[0]

            # Create color box with padding
            color_box = QLabel()
            color_box.setFixedSize(20, 20)
            color_box.setObjectName("ColorBox")
            color_box.setStyleSheet(f'background-color: rgb({r}, {g}, {b});')

            # Create text label
            text_label = QLabel(self.class_labels[class_id])
            text_label.setStyleSheet("font-weight: bold;")
            text_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)

            # Create horizontal layout
            h_layout = QHBoxLayout()
            h_layout.setContentsMargins(5, 5, 5, 5)
            h_layout.setSpacing(10)
            h_layout.addWidget(color_box, alignment=Qt.AlignVCenter)
            h_layout.addWidget(text_label, alignment=Qt.AlignVCenter)
            h_layout.addStretch()

            # Create a QPushButton to make the legend item clickable
            legend_button = QPushButton()
            legend_button.setLayout(h_layout)
            legend_button.setStyleSheet("""
                QPushButton {
                    text-align: left;
                    background-color: transparent;
                    border: none;
                }
                QPushButton:hover {
                    background-color: #f0f0f0;
                }
            """)
            legend_button.setCursor(Qt.PointingHandCursor)
            legend_button.setFlat(True)  # Remove button borders

            # Connect the button's clicked signal with a lambda to pass class_id
            legend_button.clicked.connect(lambda checked, cid=class_id: self.highlightClass(cid))

            self.legendLayout.addWidget(legend_button)

    def highlightClass(self, class_id):
        """
        Highlights bounding boxes of the selected class. If the same class is clicked again, removes the highlight.
        """
        if self.processing_video:
            return  # Skip highlighting when processing video

        if self.current_highlight_class == class_id:
            # If already highlighted, remove highlighting
            self.current_highlight_class = None
        else:
            self.current_highlight_class = class_id
        self.displayDetectedImage(highlight_class_id=self.current_highlight_class)

        # Provide visual feedback for the selected legend item
        for i in range(self.legendLayout.count()):
            widget = self.legendLayout.itemAt(i).widget()
            if isinstance(widget, QPushButton):
                # Reset all buttons to default style
                widget.setStyleSheet("""
                    QPushButton {
                        text-align: left;
                        background-color: transparent;
                        border: none;
                    }
                    QPushButton:hover {
                        background-color: #f0f0f0;
                    }
                """)
                # Highlight the selected button
                if self.current_highlight_class is not None:
                    # Find the label inside the button
                    labels = widget.findChildren(QLabel)
                    for label in labels:
                        if label.text() == self.class_labels[class_id]:
                            widget.setStyleSheet("""
                                QPushButton {
                                    text-align: left;
                                    background-color: #d0e0ff;
                                    border: none;
                                }
                                QPushButton:hover {
                                    background-color: #b0c0ff;
                                }
                            """)

    def clearLayout(self, layout):
        """
        Recursively clears a layout by removing all widgets and sub-layouts.
        """
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self.clearLayout(item.layout())

    def resizeEvent(self, event):
        """
        Override the resizeEvent to ensure the image is rescaled appropriately when the window is resized.
        """
        super().resizeEvent(event)
        self.rescaleImage()

    def uploadVideo(self):
        self.processing_video = True  # We're processing a video
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            self, 'Select Video', '', 'Videos (*.mp4 *.avi *.mov)', options=options)
        if fileName:
            self.videoPath = fileName
            self.processVideo(self.videoPath)
            # Switch to video display
            self.displayStack.setCurrentWidget(self.videoWidget)
        else:
            self.statusBar.showMessage("Video upload canceled.", 3000)

    def processVideo(self, videoPath):
        self.statusBar.showMessage("Processing video, please wait...")
        # Disable the upload buttons
        self.uploadButton.setEnabled(False)
        self.uploadVideoButton.setEnabled(False)

        # Create a QThread
        self.thread = QThread()
        # Create the worker and move it to the thread
        self.worker = VideoProcessingWorker(
            videoPath,
            self.yolo_model,
            self.class_labels,
            self.class_colors
        )
        self.worker.moveToThread(self.thread)
        # Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.videoProcessingFinished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        # Start the thread
        self.thread.start()

    def videoProcessingFinished(self, output_path):
        self.statusBar.showMessage("Video processing completed.", 5000)
        # Enable the upload buttons
        self.uploadButton.setEnabled(True)
        self.uploadVideoButton.setEnabled(True)
        self.processed_video_path = output_path
        self.playButton.setEnabled(True)
        # Now, play the processed video
        self.playVideo(self.processed_video_path)

    def playVideo(self, videoPath):
        # Switch to video widget
        self.displayStack.setCurrentWidget(self.videoWidget)

        # Load the video into the media player
        url = QUrl.fromLocalFile(os.path.abspath(videoPath))
        if not url.isValid():
            self.statusBar.showMessage("Invalid video URL.", 5000)
            return

        self.mediaPlayer.setMedia(QMediaContent(url))
        self.mediaPlayer.play()
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        else:
            self.mediaPlayer.play()
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def positionChanged(self, position):
        self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    def handleMediaError(self):
        error_string = self.mediaPlayer.errorString()
        QMessageBox.critical(self, "Media Player Error", f"Error: {error_string}")
        self.statusBar.showMessage(f"Media Player Error: {error_string}", 5000)

    def closeEvent(self, event):
        # Stop the media player
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.stop()
        # Clean up temporary files
        if self.processed_video_path and os.path.exists(self.processed_video_path):
            os.remove(self.processed_video_path)
        event.accept()

class VideoProcessingWorker(QObject):
    finished = pyqtSignal(str)  # Emit the output video path when done

    def __init__(self, videoPath, yolo_model, class_labels, class_colors):
        super().__init__()
        self.videoPath = videoPath
        self.yolo_model = yolo_model
        self.class_labels = class_labels
        self.class_colors = class_colors

    def run(self):
        # Open video file
        cap = cv2.VideoCapture(self.videoPath)

        if not cap.isOpened():
            # Emit a signal to indicate failure (you can expand this to handle errors)
            self.finished.emit('')
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create a VideoWriter to write the processed video to a temporary file
        # Use a fixed path for testing
        temp_output_path = 'processed_video.avi'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

        # Process frames
        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1
            frame_skip = 3
            if frame_number % frame_skip != 0:
                continue

            # Perform detection on the frame
            detections = self.detectObjectsInFrame(frame)

            # Draw detections on the frame
            frame = self.drawDetectionsOnFrame(frame, detections)

            # Write the frame to output video
            out.write(frame)

            # Optional: Keep UI responsive
            QApplication.processEvents()

        # Release resources
        cap.release()
        out.release()

        # Emit signal with the output path
        self.finished.emit(temp_output_path)

    def detectObjectsInFrame(self, frame):
        # Similar to performDetection(), but operates on a single frame
        test_img = frame.copy()
        scalefactor = 1.0 / 255.0
        new_size = (416, 416)
        blob = cv2.dnn.blobFromImage(test_img, scalefactor, new_size, swapRB=True, crop=False)
        self.yolo_model.setInput(blob)
        obj_detections_in_layers = self.yolo_model.forward(self.output_layers())

        img_height, img_width = test_img.shape[:2]
        class_ids = []
        confidences = []
        boxes = []

        for layer in obj_detections_in_layers:
            for detection in layer:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * img_width)
                    center_y = int(detection[1] * img_height)
                    width = int(detection[2] * img_width)
                    height = int(detection[3] * img_height)
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)
                    boxes.append([x, y, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        detections = []

        if isinstance(indices, (list, np.ndarray)) and len(indices) > 0:
            if isinstance(indices, tuple):
                indices = indices[0]
            try:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    class_id = class_ids[i]
                    confidence = confidences[i]
                    color = self.class_colors[class_id]
                    detection = {
                        'class_id': class_id,
                        'box': [x, y, w, h],
                        'confidence': confidence,
                        'color': color
                    }
                    detections.append(detection)
            except AttributeError:
                detections = []
        else:
            detections = []

        return detections

    def drawDetectionsOnFrame(self, frame, detections):
        for detection in detections:
            class_id = detection['class_id']
            x, y, w, h = detection['box']
            color = detection['color']
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f'{self.class_labels[class_id]}: {detection["confidence"]:.2f}'
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
        return frame

    def output_layers(self):
        model_layers = self.yolo_model.getLayerNames()
        return [model_layers[i - 1] for i in self.yolo_model.getUnconnectedOutLayers()]

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = YOLOv4App()
    ex.show()
    sys.exit(app.exec_())
