# BNO055 IMU Data Logger

This repository contains Arduino and Python scripts for logging acceleration and angular velocity data from a BNO055 Inertial Measurement Unit (IMU) to a CSV file.

## Overview

The system consists of two main parts:

1.  **Arduino Sketch**: Runs on an ESP32 (or compatible microcontroller) connected to a BNO055 sensor. It reads linear acceleration and angular velocity data and sends it over a serial port.
2.  **Python Script**: Runs on a computer, connects to the serial port, reads the data sent by the Arduino, and saves it into a CSV file.

## Features

*   **Sensor Data Acquisition**: Captures 3-axis linear acceleration and 3-axis angular velocity from the BNO055 IMU.
*   **Serial Communication**: Transmits sensor data from the microcontroller to the computer via serial.
*   **CSV Logging**: Stores the acquired data in a timestamped CSV file for easy analysis.
*   **Configurable Sample Rate**: The Arduino sketch allows setting the data sampling rate.

## Hardware Requirements

*   **BNO055 IMU Sensor**: Connected via I2C.
*   **ESP32 Development Board**: Or any Arduino-compatible microcontroller with serial communication capabilities.
*   **USB Cable**: For connecting the microcontroller to the computer.

## Software Requirements

*   **Arduino IDE**: For uploading the sketch to the microcontroller.
    *   Libraries: `Wire`, `Adafruit_Sensor`, `Adafruit_BNO055`
*   **Python 3**: For running the data logging script.
    *   Libraries: `pyserial`

## Setup Guide

### 1. Arduino (Microcontroller) Setup

1.  **Install Arduino IDE**: If you don't have it, download and install the Arduino IDE.
2.  **Install Libraries**:
    *   Open Arduino IDE, go to `Sketch > Include Library > Manage Libraries...`
    *   Search for and install:
        *   `Adafruit BNO055`
        *   `Adafruit Unified Sensor`
3.  **Connect BNO055 to ESP32**:
    *   **VIN** to **3.3V** (or 5V, check your BNO055 module's voltage tolerance)
    *   **GND** to **GND**
    *   **SDA** to **SDA** (typically GPIO21 on ESP32)
    *   **SCL** to **SCL** (typically GPIO22 on ESP32)
    *   **ADDR** pin: The default I2C address for the BNO055 is `0x28` when the ADDR pin is connected to GND, or `0x29` when connected to VIN. The provided sketch uses `0x28`. Adjust `Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28, &Wire);` if your sensor uses `0x29`.
4.  **Open and Upload Arduino Sketch**:
    *   Open the `bno055_arduino_logger.ino` sketch in the Arduino IDE.
    *   Select your ESP32 board (`Tools > Board > ESP32 Arduino > ...`) and the correct serial port (`Tools > Port`).
    *   Upload the sketch to the ESP32.

### 2. Python (Computer) Setup

1.  **Install Python**: Ensure Python 3 is installed on your computer.
2.  **Install `pyserial`**:
    ```bash
    pip install pyserial
    ```
3.  **Configure Python Script**:
    *   Open the `bno055_python_logger.py` script.
    *   Modify `PORT = "COM5"` to match the serial port of your ESP32 (e.g., `/dev/ttyUSB0` on Linux, `COMx` on Windows).
    *   Optionally, change `BAUDRATE = 115200` if you modify it in the Arduino sketch (though 115200 is standard).
    *   Optionally, change `SAVE_FOLDER = "./"` to your desired directory for saving CSV files.
4.  **Run Python Script**:
    ```bash
    python bno055_python_logger.py
    ```
    The script will start logging data to a CSV file named `BNO055Data_YYYYMMDD_HHMMSS.csv` in the specified `SAVE_FOLDER`. Press `Ctrl+C` to stop logging.

## Data Format

The generated CSV file will have the following columns:

`accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z`

*   `accel_x, accel_y, accel_z`: Linear acceleration in X, Y, Z axes (m/s²).
*   `gyro_x, gyro_y, gyro_z`: Angular velocity in X, Y, Z axes (rad/s).

## Troubleshooting

*   **"Ooops, no BNO055 detected..."**:
    *   Check your wiring carefully, especially for SDA, SCL, VIN, and GND.
    *   Ensure the I2C address in the Arduino sketch (`0x28` or `0x29`) matches your BNO055 module's configuration.
    *   Verify the BNO055 is powered correctly.
*   **Python script doesn't connect**:
    *   Make sure the `PORT` variable in the Python script is correct for your microcontroller.
    *   Ensure the Arduino serial monitor is closed, as only one application can access the serial port at a time.
    *   Check if the ESP32 is powered on and the Arduino sketch is running.

## Future Enhancements

*   Add timestamps to each data entry.
*   Include other BNO055 sensor data (e.g., quaternions, Euler angles, magnetometer).
*   Implement error handling for serial communication.
*   Graphical real-time data visualization.
*   Integration with machine learning libraries for motion analysis.
