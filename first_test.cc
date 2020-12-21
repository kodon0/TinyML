// INCLUDING DEPENDENCIES

#include "tensorflow/lite/micro/examples/hello_world/sine_model_data.h"
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// SETTING UP THE TEST

TF_LITE_MICRO_TESTS_BEGIN
TF_LITE_MICRO_TEST(LoadModelAndPerformInference) {

// MicroErrorReporter overrides a mthod on ErrorReporter
// Create an ErrorReporter and point ot to micro_error_reporter
// A pointer doesnt hold a variable,(*) it holds a place in memory where the value exists
// User error_reporter for debugging

// GETTING READY TO LOG DATA

tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

// Map the model into a usable data structure. This doesn't involve any
// copying or parsing, it's a very lightweight operation.

// Get data array defined in simple_model, which returns a model pointer
// This gets assigned to variable model == simple_model
// The Model is a struct which is similar to a class in C++

// MAPPING CREATED MODEL

const tflite::Model* model = ::tflite::GetModel(g_sine_model_data);

// Call model version number and compare it to TFLITE_SCHEMA_VERSION in tfline library used
// Check if numbers match. -> is the arrow operator

if (model->version() != TFLITE_SCHEMA_VERSION) { error_reporter->Report(
"Model provided is schema version %d not equal " "to supported version %d.\n",
model->version(), TFLITE_SCHEMA_VERSION); return 1;
}

// Carry on even if versions don't match

error_reporter->Report(
"Model provided is schema version %d not equal " "to supported version %d.\n",
model->version(), TFLITE_SCHEMA_VERSION);

// Next up, we create an instance of AllOpsResolver:
// This pulls in all the operation implementations we need.
// This class allows TFL interpreter to access operations
// Allows access to operations to  turn inputs to outputs

// CREATING ALLOPSRESOLVER

tflite::ops::micro::AllOpsResolver resolver;

// DEFINING TENSOR AREA

// Create an area of memory to use for input, output, and intermediate arrays.
// Finding the minimum value for your model may require some trial and error.

const int tensor_arena_size = 2 × 1024;
uint8_t tensor_arena[tensor_arena_size];

// CREATING AN INTERPRETER
// Build an interpreter to run the model

tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, tensor_arena_size, error_reporter);
// Allocate memory from the tensor_arena for the model's tensors

interpreter.AllocateTensors();

// Obtain a pointer to the model's input tensor
// Model can have many tensors so need to specify which. We want first once

// CHECKING TENSOR INPUT

TfLiteTensor* input = interpreter.input(0);


// This next chunk allows for checks or assertions (NE = not equal, EQ = equal, GT = greater than etc.)
// nullptr represents a null pointer which doesnt point. i.e, checking if we actually HAVE and input!
TF_LITE_MICRO_EXPECT_NE(nullptr, input);
// The property "dims" tells us the tensor's shape. It has one element for
// each dimension. Our input is a 2D tensor containing 1 element, so "dims"
// should have size 2.
// In keras: 0 is actually [[0]]
TF_LITE_MICRO_EXPECT_EQ(2, input->dims->size);
// The value of each element gives the length of the corresponding tensor.
// We should expect two single element tensors (one is contained within the // other).
TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[1]);
// Check the input is a 32 bit floating point value
TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, input->type);

// RUNNING INFERENCE ON AN INPUT

// Provide an input value
input->data.f[0] = 0.; // data var is a union, allowing to store multiple data types in one spot in memory
// Assing floating point to first location in allocated memory
// Run the model on this input and check that it succeeds
TfLiteStatus invoke_status = interpreter.Invoke();
if (invoke_status != kTfLiteOk) {
error_reporter->Report("Invoke failed\n");
}
TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

// READING THE OUTPUT

TfLiteTensor* output = interpreter.output(0);

// Output, like input is a floating scalar in a 2D tensors

// Do the same checks as before
TF_LITE_MICRO_EXPECT_EQ(2, output->dims->size);
TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[1]);
TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, output->type);

// Obtain the output value from the tensor
float value = output->data.f[0];

// Check that the output value is within 0.05 of the expected value. If pass, things are good
TF_LITE_MICRO_EXPECT_NEAR(0., value, 0.05);
