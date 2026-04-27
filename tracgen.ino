#include <Servo.h>
#include <Wire.h>
#include <math.h>
#include <Adafruit_INA219.h>

Servo s1, s2, s3, s4, s5;
const int sPIN1 = 9, sPIN2 = 6, sPIN3 = 5, sPIN4 = 3, sPIN5 = 11;

Adafruit_INA219 ina219;

// ---------------------------
// INA219 logging state
// ---------------------------
bool inaLogging = false;
unsigned long inaLastSampleMs = 0;
const unsigned long INA_SAMPLE_PERIOD_MS = 2;

unsigned long inaSampleCount = 0;
float inaCurrentSum_mA = 0.0;
float inaPeak_mA = 0.0;
float inaMinBus_V = 999.0;

unsigned long inaStartMs = 0;
unsigned long inaFirstOverMs = 0;
unsigned long inaPeakMs = 0;
bool inaFirstOverSeen = false;
const float INA_TRIGGER_THRESHOLD_MA = 100.0;

// ---------------------------
// Helpers
// ---------------------------
void resetINALogStats() {
  inaSampleCount = 0;
  inaCurrentSum_mA = 0.0;
  inaPeak_mA = 0.0;
  inaMinBus_V = 999.0;

  inaStartMs = 0;
  inaFirstOverMs = 0;
  inaPeakMs = 0;
  inaFirstOverSeen = false;
}

void startINALogging() {
  resetINALogStats();
  inaLogging = true;
  inaLastSampleMs = millis();
  inaStartMs = millis();
  Serial.println("INA219_LOGGING_START");
}

void stopINALoggingAndPrintSummary() {
  inaLogging = false;

  float avg_mA = 0.0;
  if (inaSampleCount > 0) {
    avg_mA = inaCurrentSum_mA / (float)inaSampleCount;
  } else {
    inaMinBus_V = 0.0;
  }

  Serial.print("INA219_SUMMARY");
  Serial.print(",samples=");
  Serial.print(inaSampleCount);
  Serial.print(",avg_mA=");
  Serial.print(avg_mA, 2);
  Serial.print(",peak_mA=");
  Serial.print(inaPeak_mA, 2);
  Serial.print(",min_bus_V=");
  Serial.print(inaMinBus_V, 3);
  Serial.print(",first_over_mA_ms=");
  Serial.print(inaFirstOverSeen ? inaFirstOverMs : 0);
  Serial.print(",peak_ms=");
  Serial.println(inaPeakMs);
}

void updateINALogger() {
  if (!inaLogging) return;

  unsigned long now = millis();
  if (now - inaLastSampleMs < INA_SAMPLE_PERIOD_MS) return;
  inaLastSampleMs = now;

  float current_mA_signed = ina219.getCurrent_mA();
  float current_mA = fabs(current_mA_signed);   // use magnitude
  float bus_V = ina219.getBusVoltage_V();

  unsigned long relMs = now - inaStartMs;

  inaSampleCount++;
  inaCurrentSum_mA += current_mA;

  if (!inaFirstOverSeen && current_mA >= INA_TRIGGER_THRESHOLD_MA) {
    inaFirstOverSeen = true;
    inaFirstOverMs = relMs;
  }

  if (inaSampleCount == 1 || current_mA > inaPeak_mA) {
    inaPeak_mA = current_mA;
    inaPeakMs = relMs;
  }

  if (inaSampleCount == 1 || bus_V < inaMinBus_V) {
    inaMinBus_V = bus_V;
  }
}

void handleServoCommand(const String& line) {
  float a1, a2, a3, a5 = 45.0;

  int c1 = line.indexOf(',');
  int c2 = line.indexOf(',', c1 + 1);
  if (c1 < 0 || c2 < 0) return;

  int c3 = line.indexOf(',', c2 + 1);

  a1 = line.substring(0, c1).toFloat();
  a2 = line.substring(c1 + 1, c2).toFloat();

  if (c3 > 0) {
    a3 = line.substring(c2 + 1, c3).toFloat();
    a5 = line.substring(c3 + 1).toFloat();
  } else {
    a3 = line.substring(c2 + 1).toFloat();
  }

  a1 = constrain(a1, 0, 180);
  a2 = constrain(a2, 0, 180);
  a3 = constrain(a3, 0, 180);
  a5 = constrain(a5, 0, 180);

  s1.write((int)a1);
  s2.write((int)a2);
  s3.write((int)a3);

  if (c3 > 0) {
    s5.write((int)a5);
  }
}

void handleSerialLine(String line) {
  line.trim();
  if (line.length() == 0) return;

  // INA commands
  if (line.equalsIgnoreCase("INA_START")) {
    startINALogging();
    return;
  }

  if (line.equalsIgnoreCase("INA_STOP")) {
    stopINALoggingAndPrintSummary();
    return;
  }

  handleServoCommand(line);
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(10);

  s1.attach(sPIN1);
  s2.attach(sPIN2);
  s3.attach(sPIN3);
  s4.attach(sPIN4);
  s5.attach(sPIN5);

  s1.write(90);
  s2.write(37);
  s3.write(180);
  s4.write(66);
  s5.write(45);

  Wire.begin();

  if (!ina219.begin()) {
    Serial.println("INA219_INIT_FAIL");
  } else {
    Serial.println("INA219_INIT_OK");
  }

  resetINALogStats();

  Serial.println("ARDUINO_READY");
}

void loop() {
  // Always keep sampling INA219 when enabled
  updateINALogger();

  // Only parse serial when a full line is available
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    handleSerialLine(line);
  }
}