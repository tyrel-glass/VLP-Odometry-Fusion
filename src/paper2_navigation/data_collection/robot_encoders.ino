//=========== Motors ===========
#define ENCA 2         // Encoder pin A
#define ENCB 3         // Encoder pin B

// Motor A
#define PWMA 5
#define AIN2 6
#define AIN1 7

#define STBY 8

#define BIN1 9
#define BIN2 10
#define PWMB 11

int pwm = 100;         // Motor speed

volatile int encoderCountA = 0;
volatile int encoderCountB = 0;

float counts_per_degree = 10.0;  // Adjust this factor based on your hardware

//=========== Interrupt Handlers ===========
void encoderA_ISR() {
  bool stateA = digitalRead(ENCA);
  bool stateB = digitalRead(ENCB);

  if (stateA == stateB) {
    encoderCountA++;
  } else {
    encoderCountA--;
  }
}

void encoderB_ISR() {
  bool stateA = digitalRead(ENCA);
  bool stateB = digitalRead(ENCB);

  if (stateA == stateB) {
    encoderCountB++;
  } else {
    encoderCountB--;
  }
}

//=========================================================================
// SETUP
//=========================================================================

void setup() {
  Serial.begin(9600);

  // Motor Pins
  pinMode(STBY, OUTPUT);
  pinMode(PWMA, OUTPUT);
  pinMode(AIN1, OUTPUT);
  pinMode(AIN2, OUTPUT);
  pinMode(PWMB, OUTPUT);
  pinMode(BIN1, OUTPUT);
  pinMode(BIN2, OUTPUT);

  // Encoder Pins
  pinMode(ENCA, INPUT);
  pinMode(ENCB, INPUT);

  // Attach Interrupts
  attachInterrupt(digitalPinToInterrupt(ENCA), encoderA_ISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENCB), encoderB_ISR, CHANGE);

  Serial.println("SETUP COMPLETE");
}

//=========================================================================
// MAIN LOOP
//=========================================================================

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    Serial.print("received command: ");
    Serial.println(command);
    processCommand(command);
  }
}

//=========================================================================
// SERIAL COMMAND PROCESSING
//=========================================================================

void processCommand(String command) {
  command.trim();  // Remove leading/trailing whitespace and newline characters

  if (command.equals("test_encoder")) {
    testEncoder();
  } else {
    int commaIndex = command.indexOf(',');
    if (commaIndex != -1) {
      String txt = command.substring(0, commaIndex);
      float value = command.substring(commaIndex + 1).toFloat();

      if (txt.equals("rotate")) {
        rotate(value);  // Positive for right, negative for left
      } else if (txt.equals("forward")) {
        forward_distance(value);
      } else {
        Serial.println("Unknown command.");
      }
    } else {
      Serial.println("Invalid command format.");
    }
  }
}

//=========================================================================
// TEST FUNCTIONALITY
//=========================================================================

void testEncoder() {
  Serial.println("Testing encoder functionality...");
  encoderCountA = 0;  // Reset encoder counts
  encoderCountB = 0;


  // Serial.println("moving forward");
  // move(1, pwm, 1);  // Motor 1 backward
  // move(2, pwm, 1);  // Motor 2 forward


  unsigned long startTime = millis();
  while (millis() - startTime < 5000) { // Test for 5 seconds
    Serial.print("Encoder A Count: ");
    Serial.print(encoderCountA);
    Serial.print("\tEncoder B Count: ");
    Serial.println(encoderCountB);
    delay(500); // Print every 500 ms
  }


  // Serial.println("moving backwards");
  // move(1, pwm, 0);  // Motor 1 backward
  // move(2, pwm, 0);  // Motor 2 forward

  startTime = millis();
  while (millis() - startTime < 5000) { // Test for 5 seconds
    Serial.print("Encoder A Count: ");
    Serial.print(encoderCountA);
    Serial.print("\tEncoder B Count: ");
    Serial.println(encoderCountB);
    delay(500); // Print every 500 ms
  }

  stop();

  Serial.println("Encoder test complete.");
}

//=========================================================================
// MOVEMENT CONTROL
//=========================================================================

void rotate(float degrees) {
  int target_count = abs(degrees) * counts_per_degree;  // Total encoder counts to achieve the desired rotation

  int start_pos = encoderCountA;  // Use encoder A for rotation tracking

  if (degrees > 0) {
    // Rotate Right
    move(1, pwm, 1);  // Motor 1 backward
    move(2, pwm, 0);  // Motor 2 forward
    while (encoderCountA < start_pos + target_count) {
      // Wait until rotation is complete
    }
  } else {
    // Rotate Left
    move(1, pwm, 0);  // Motor 1 forward
    move(2, pwm, 1);  // Motor 2 backward
    while (encoderCountA > start_pos - target_count) {
      // Wait until rotation is complete
    }
  }

  stop();  // Stop motors after rotation
}

void forward_distance(float dist) {
  float counts_per_mm = 1.0;  // Adjust this factor based on your hardware
  int target_count = dist * counts_per_mm;

  int start_pos = encoderCountB;

  forward();

  while (encoderCountB < start_pos + target_count) {
    // Wait until the distance is covered
  }
  stop();
}

//=========================================================================
// MOTOR CONTROL
//=========================================================================

void move(int motor, int speed, int direction) {
  digitalWrite(STBY, HIGH); // Disable standby

  boolean inPin1 = LOW;
  boolean inPin2 = HIGH;

  if (direction == 1) {
    inPin1 = HIGH;
    inPin2 = LOW;
  }

  if (motor == 1) {
    digitalWrite(AIN1, inPin1);
    digitalWrite(AIN2, inPin2);
    analogWrite(PWMA, speed);
  } else {
    digitalWrite(BIN1, inPin1);
    digitalWrite(BIN2, inPin2);
    analogWrite(PWMB, speed);
  }
}

void forward() {
  move(1, pwm, 0);
  move(2, pwm, 0);
}

void stop() {
  digitalWrite(STBY, LOW);  // Stop the motors
}
