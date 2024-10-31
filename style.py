# Define a stylesheet for the application
STYLE = """
QWidget {
    background-color: #2E3440; /* Dark background */
    color: #D8DEE9; /* Light text */
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

QPushButton {
    background-color: #4C566A;
    border: none;
    color: #ECEFF4;
    padding: 10px 20px;
    border-radius: 5px;
    font-size: 14px;

}

QPushButton:hover {
    background-color: #5E81AC;
}

QPushButton:pressed {
    background-color: #81A1C1;
}

QLabel#TitleLabel {
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 20px;
    color: #88C0D0; /* Accent color */
}

QScrollArea {
    background-color: #3B4252;
    border-radius: 5px;
    padding: 10px;
}

QScrollArea QWidget {
    background-color: #3B4252;
}

QLabel {
    font-size: 14px;
}

/* Image Display */
QLabel#ImageLabel {
    border: 2px solid #4C566A;
    border-radius: 10px;
    padding: 5px;
    background-color: #3B4252;
}

/* Legend Colors */
QLabel#ColorBox {
    border: 1px solid #ECEFF4;
    border-radius: 3px;
}
"""
