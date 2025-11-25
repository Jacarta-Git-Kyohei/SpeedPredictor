import asyncio
import csv
import os
from bleak import BleakClient, BleakScanner
from datetime import datetime

# ------ Initial Parameter ------ #
SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
CHARACTERISTIC_UUID = "12345678-1234-5678-1234-56789abcdef1"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE_SAVED_FOLDER = os.path.join(SCRIPT_DIR, "..", "RawData")
PROGRAM_DELAY_TIME = 15
# ------------------------------- #

async def main():
    print("Scanning for BLE devices...")
    devices = await BleakScanner.discover()
    target = None

    for d in devices:
        if d.name and "BNO055_BLE_Sensor" in d.name:
            target = d
            break

    if target is None:
        print("Device not found.")
        print("=== Available Devices ===")
        for d in devices:
            print(f"- {d.name} ({d.address})")
        print("=========================")
        return

    print(f"Connecting to {target.name} ({target.address})...")
    async with BleakClient(target.address) as client:
        print("Connected:", client.is_connected)

        # 30 Second Delay
        print(f"Waiting for {PROGRAM_DELAY_TIME} seconds before starting data logging...")
        await asyncio.sleep(PROGRAM_DELAY_TIME)
        print("Starting data logging!")

        timestamp = datetime.now().strftime("%m%d_%H%M")
        filename = os.path.join(OUTPUT_FILE_SAVED_FOLDER, f"RawData_{timestamp}.csv")

        # If the folder does not exist, create it.
        os.makedirs(OUTPUT_FILE_SAVED_FOLDER, exist_ok=True)

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"])

            def notification_handler(sender, data):
                try:
                    line = data.decode("utf-8").strip()
                    values = line.split(",")
                    if len(values) == 7:
                        writer.writerow(values)
                        print(values)
                except Exception as e:
                    print("Error decoding:", e)

            await client.start_notify(CHARACTERISTIC_UUID, notification_handler)
            print("Receiving data... Press Ctrl+C to stop.")
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping...")
            finally:
                await client.stop_notify(CHARACTERISTIC_UUID)

if __name__ == "__main__":
    asyncio.run(main())
