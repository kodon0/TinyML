// Since only have an LED on arduino, and values for hello-wolrd vary betwee -1 and 1
// Can make 0 LED fully off and -1 AND +1 as fully lit LED. This will show a linking pattern

// kInferencesPerCycle can be adjusted to amount of inferences in constants.cc -> speed of LED
// Specific arduion version in hello_world/arduino/constants.cc.

// Use PWM (pulse width modulation) to dim the LED
//Found here hello_world/arduino/ output_handler.cc

// old is_initialized -> new initialized, removed static

// IMPORTS

#include "tensorflow/lite/micro/examples/hello_world/output_handler.h"

#include "Arduino.h" //for arduino platform -- use to control board
#include "tensorflow/lite/micro/examples/hello_world/constants.h" //use for kInferencesPerCycle

// The pin of the Arduino's built-in LED
int led = LED_BUILTIN;

// Track whether the function has run at least once
bool initialized = false;

// Animates a dot across the screen to represent the current x and y values
void HandleOutput(tflite::ErrorReporter* error_reporter, float x_value,
                  float y_value) {
  // Do this only once
  if (!initialized) {
    // Set the LED pin to output
    pinMode(led, OUTPUT);
    initialized = true;
  }

  // initialized to true, so the above wont run again

  // Calculate the brightness of the LED such that y = -1 is fully off
  // and y = 1 is fully on. The LED's brightness can range from 0-255.
  int brightness = (int)(127.5f * (y_value + 1));
//
// Setting LED brightness accoridng to defined brightness limits
  // Set the brightness of the LED. If the specified pin does not support PWM,
  // this will result in the LED being on when y > 127, off otherwise.
  analogWrite(led, brightness);
  // Built-in analogueWrite() takes in led pin (= LED_BUILTIN) and brightness

  // pinMODE section initialises the pin in arduino as an OUTPUT (vs input)


// In old version this is error_reporter->Report("%d\n", brightness)

  // Log the current brightness value for display in the Arduino plotter
  TF_LITE_REPORT_ERROR(error_reporter, "%d\n", brightness);
}

// On the Arduino platform, the ErrorReporter is set up to log data via a serial port.
// Serial is a very common way for microcontrollers to communicate with host computers,
// and it’s often used for debugging.t’s a communication protocol in which data is communicated
//one bit at a time by switching an output pin on and off. We can use it to send and receive anything,
// from raw binary data to text and numbers. - from author

// Will use Serial Plotter in Arduino IDE to plot answers
// Arduino needs files to end in .cpp, not .cc
// setup() and loop() are called automatically, voiding need for main.cc and main()
