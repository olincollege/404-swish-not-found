#include <AccelStepper.h>
#include <MultiStepper.h>

// Define stepper motors (DIR, STEP)
AccelStepper stepper1(AccelStepper::DRIVER, 3, 2);  // Bottom-left motor
AccelStepper stepper2(AccelStepper::DRIVER, 5, 4);  // Bottom-right motor
AccelStepper stepper3(AccelStepper::DRIVER, 9, 8);  // Top-right motor
AccelStepper stepper4(AccelStepper::DRIVER, 7, 6);  // Top-left motor

MultiStepper steppers;

// Constants for motor and frame setup
const int stepsPerInch = 130;  // Steps per inch of cable movement
const float frameSizeX = 34.00;
const float frameSizeY = 29.50;
const float boardSizeX = 9.0;
const float boardSizeY = 7.0;
const float halfBoardX = boardSizeX / 2;  // Half board size for calculations
const float halfBoardY = boardSizeY / 2;  // Half board size for calculations
float targetX;
float targetY;
// Center position in inches
const float centerX = frameSizeX / 2;
const float centerY = frameSizeY / 2;


// Motor step targets
long motorSteps[4];

void setup() {
  Serial.begin(115200);

  // Add steppers to MultiStepper
  steppers.addStepper(stepper1);
  steppers.addStepper(stepper2);
  steppers.addStepper(stepper3);
  steppers.addStepper(stepper4);

  // Set max speed for each motor
  int maxSpeed = 3000;
  stepper1.setMaxSpeed(maxSpeed);
  stepper2.setMaxSpeed(maxSpeed);
  stepper3.setMaxSpeed(maxSpeed);
  stepper4.setMaxSpeed(maxSpeed);

  stepper1.setCurrentPosition(-16.817 * stepsPerInch);
  stepper2.setCurrentPosition(16.817 * stepsPerInch);
  stepper3.setCurrentPosition(16.817 * stepsPerInch);
  stepper4.setCurrentPosition(-16.817 * stepsPerInch);

  // stepper1.setCurrentPosition(-3.5 * stepsPerInch);
  // stepper2.setCurrentPosition(21.5 * stepsPerInch);
  // stepper3.setCurrentPosition(28.5 * stepsPerInch);
  // stepper4.setCurrentPosition(-19.5 * stepsPerInch);

  Serial.println("Motors zeroed.");
}

void loop() {
  static bool waitingForX = true;
  static float targetX = 0.0;
  static float targetY = 0.0;

  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');  // Read data until newline
    int commaIndex = input.indexOf(',');          // Find the position of the comma

    if (commaIndex != -1) {  // Ensure the data format is correct
      String value1_str = input.substring(0, commaIndex);
      String value2_str = input.substring(commaIndex + 1);

      targetX = value1_str.toInt();  // Convert to integer
      targetY = value2_str.toInt();  // Convert to integer

      // Print received values
      Serial.print("Received: ");
      Serial.print(targetX);
      Serial.print(" and ");
      Serial.println(targetY);

      int bandLen = 1;
      targetX = constrain(targetX, halfBoardX + bandLen, frameSizeX - halfBoardX - bandLen);
      targetY = constrain(targetY, halfBoardY + bandLen, frameSizeY - halfBoardY - bandLen);

      // Calculate motor steps based on target coordinates
      calculateLengths(targetX, targetY);

      Serial.println(motorSteps[0]/stepsPerInch);
      Serial.println(motorSteps[1]/stepsPerInch);
      Serial.println(motorSteps[2]/stepsPerInch);
      Serial.println(motorSteps[3]/stepsPerInch);

      // // Move motors to calculated positions
      steppers.moveTo(motorSteps);
      steppers.runSpeedToPosition();

      // // Wait 2 seconds at target position
      delay(2000);

      // // Return to the center position
      moveToCenter();
    }
  }
}

void calculateLengths(float targetX, float targetY) {
  motorSteps[0] = -round(sqrt(sq(targetX - halfBoardX) + sq(targetY - halfBoardY)) * stepsPerInch);                           // Bottom-left motor
  motorSteps[1] = round(sqrt(sq(frameSizeX - targetX - halfBoardX) + sq(targetY - halfBoardY)) * stepsPerInch);               // Bottom-right motor
  motorSteps[2] = round(sqrt(sq(frameSizeX - targetX - halfBoardX) + sq(frameSizeY - targetY - halfBoardY)) * stepsPerInch);  // Top-right motor
  motorSteps[3] = -round(sqrt(sq(targetX - halfBoardX) + sq(frameSizeY - targetY - halfBoardY)) * stepsPerInch);              // Top-left motor
}

// Function to move to the center position
void moveToCenter() {
  calculateLengths(centerX, centerY);
  steppers.moveTo(motorSteps);
  steppers.runSpeedToPosition();
}