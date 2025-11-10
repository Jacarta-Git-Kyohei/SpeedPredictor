#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>

/* Set the delay between fresh samples */
/* Default is 100 Hz ( Delay is 10 ms )*/
uint16_t BNO055_SAMPLERATE_DELAY_MS = 10;

// Check I2C device address and correct line below (by default address is 0x29 or 0x28)
//                                   id, address
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28, &Wire);

void setup(void)
{
  Serial.begin(115200);

  while (!Serial) delay(10);  // wait for serial port to open!

  Serial.println("Orientation Sensor Test"); Serial.println("");

  /* Initialise the sensor */
  if (!bno.begin())
  {
    /* There was a problem detecting the BNO055 ... check your connections */
    Serial.print("Ooops, no BNO055 detected ... Check your wiring or I2C ADDR!");
    while (1);
  }

  delay(1000);

  // Display CSV header
  Serial.println("acceleration.x,acceleration.y,acceleration.z,gyro.x,gyro.y,gyro.z");
}

void loop(void)
{
  sensors_event_t angVelocityData, linearAccelData;
  bno.getEvent(&angVelocityData, Adafruit_BNO055::VECTOR_GYROSCOPE);
  bno.getEvent(&linearAccelData, Adafruit_BNO055::VECTOR_LINEARACCEL);

  Serial.print(linearAccelData.acceleration.x);
  Serial.print(",");
  Serial.print(linearAccelData.acceleration.y);
  Serial.print(",");
  Serial.print(linearAccelData.acceleration.z);
  Serial.print(",");
  Serial.print(angVelocityData.gyro.x);
  Serial.print(",");
  Serial.print(angVelocityData.gyro.y);
  Serial.print(",");
  Serial.println(angVelocityData.gyro.z);

  delay(BNO055_SAMPLERATE_DELAY_MS);
}
