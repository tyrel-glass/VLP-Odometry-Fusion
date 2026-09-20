#include <ESP8266WiFi.h>
#include <ArduinoOTA.h>

// Wi-Fi credentials
const char* ssid = "phd";
const char* password = "Jagerhouse";

// Server configuration
WiFiServer server(8080); // Port 8080 for communication

void fetchSensorReading(WiFiClient& client);
void sendStatus(WiFiClient& client, const String& statusMessage);
void sendChunkedData(WiFiClient& client, const String& data, size_t chunkSize = 50);

void setup() {
  // Initialize Serial Communication (used for sensor communication)
  Serial.begin(460800);

  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }

  // Start OTA setup
  ArduinoOTA.setHostname("ESP8266-Sensor-OTA");
  ArduinoOTA.onStart([]() {});
  ArduinoOTA.onEnd([]() {});
  ArduinoOTA.onError([](ota_error_t error) {});
  ArduinoOTA.begin();

  // Start server
  server.begin();
}

void loop() {
  // Handle OTA updates
  ArduinoOTA.handle();

  // Check for incoming client connections
  WiFiClient client = server.available();
  if (client) {
    while (client.connected()) {
      if (client.available()) {
        String command = client.readStringUntil('\n');
        command.trim();

        if (command.equalsIgnoreCase("FETCH")) {
          fetchSensorReading(client);
        } else if (command.equalsIgnoreCase("STATUS")) {
          sendStatus(client, "READY"); // Send status that the device is ready
        } else {
          client.println("Invalid Command! Use 'FETCH' or 'STATUS'.");
        }
      }
    }
    client.stop();
  }
}

void fetchSensorReading(WiFiClient& client) {
  Serial.print("s\r\n"); // Ensure sensor is off
  delay(20);

  unsigned long startTime = millis();
  String adcData = "";

  // Flush Serial input
  while (Serial.available()) {
    Serial.read();
  }

  bool commandSent = false;
  bool foundStart = false;
  bool timedOut = false;

  // Retry mechanism for sending 'b\n' command
  for (int attempt = 1; attempt <= 5; attempt++) {
    Serial.print("b\r\n"); // Send 'b' command to start sensor readings
    delay(50);

    // Wait for the start marker "[DEBUG] ADC Data:["
    startTime = millis();
    while (millis() - startTime < 300) { // Short wait for response
      if (Serial.available()) {
        char ch = Serial.read();
        adcData += ch;

        if (adcData.endsWith("[DEBUG] ADC Data:[")) {
          adcData = ""; // Clear buffer to start collecting ADC values
          foundStart = true;
          break;
        }
      }
    }

    if (foundStart) {
      commandSent = true;
      break; // Exit retry loop if data is received
    }
  }

  if (!commandSent) {
    sendStatus(client, "ERROR: Failed to send 'b' command after 5 attempts.");
    return;
  }

  // Start collecting ADC values
  startTime = millis();
  while (true) {
    if (Serial.available()) {
      char ch = Serial.read();
      if (ch == ']') {
        break; // End of ADC data
      }
      adcData += ch;
    } else if (millis() - startTime >= 2000) {
      timedOut = true;
      break; // Exit on timeout
    }
  }

  if (timedOut) {
    sendStatus(client, "ERROR: ADC data collection timed out.");
    return;
  }

  // Send ADC data in chunks
  sendChunkedData(client, adcData);

  // Send 's' command to stop the sensor
  Serial.print("s\n");

  // Clear Serial buffer
  while (Serial.available()) {
    Serial.read();
  }

  sendStatus(client, "SUCCESS: ADC data fetched successfully.");
}


void sendChunkedData(WiFiClient& client, const String& data, size_t chunkSize) {
  size_t totalSize = data.length();
  size_t numChunks = (totalSize + chunkSize - 1) / chunkSize;

  client.println("START ADC DATA"); // Indicate the start of data transmission
  for (size_t i = 0; i < numChunks; ++i) {
    String chunk = data.substring(i * chunkSize, min((i + 1) * chunkSize, totalSize));
    client.println(chunk);
    delay(10); // Prevent network congestion
  }
  client.println("END ADC DATA"); // Indicate the end of data transmission
}

void sendStatus(WiFiClient& client, const String& statusMessage) {
  client.println("STATUS: " + statusMessage);
}
