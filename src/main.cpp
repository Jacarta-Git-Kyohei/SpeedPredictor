#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

#define SERVICE_UUID        "12345678-1234-5678-1234-56789abcdef0"
#define CHARACTERISTIC_UUID "12345678-1234-5678-1234-56789abcdef1"

Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28, &Wire);
BLECharacteristic *pCharacteristic;
uint16_t BNO055_SAMPLERATE_DELAY_MS = 10;
bool deviceConnected = false;

class MyServerCallbacks : public BLEServerCallbacks
{
    void onConnect(BLEServer* pServer) override
    {
      deviceConnected = true;
    }
    void onDisconnect(BLEServer* pServer) override
    {
      deviceConnected = false;
    }
};

void setup()
{
  Serial.begin(115200);
  while (!Serial) delay(10);

  if (!bno.begin())
  {
    Serial.println("No BNO055 detected. Check wiring!");
    while (1);
  }

  BLEDevice::init("BNO055_BLE_Sensor");
  BLEServer *pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);
  pCharacteristic = pService->createCharacteristic(
                      CHARACTERISTIC_UUID,
                      BLECharacteristic::PROPERTY_NOTIFY
                    );
  pCharacteristic->addDescriptor(new BLE2902());
  pService->start();

  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  BLEDevice::startAdvertising();

  Serial.println("BLE advertising started");
}

void loop()
{
  if (deviceConnected)
  {
    unsigned long currentTime = millis();
    sensors_event_t angVelocityData, linearAccelData;
    char data[120];

    bno.getEvent(&angVelocityData, Adafruit_BNO055::VECTOR_GYROSCOPE);
    bno.getEvent(&linearAccelData, Adafruit_BNO055::VECTOR_LINEARACCEL);

    snprintf(data, sizeof(data), "%lu,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f",
             currentTime,
             linearAccelData.acceleration.x,
             linearAccelData.acceleration.y,
             linearAccelData.acceleration.z,
             angVelocityData.gyro.x,
             angVelocityData.gyro.y,
             angVelocityData.gyro.z);

    pCharacteristic->setValue((uint8_t*)data, strlen(data));
    pCharacteristic->notify();

    delay(BNO055_SAMPLERATE_DELAY_MS);
  }

  else
  {
    delay(100);
  }
}
