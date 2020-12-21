#include "tensorflow/lite/micro/examples/hello_world/sine_model_data.h"
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

TF_LITE_MICRO_TESTS_BEGIN
TF_LITE_MICRO_TEST(LoadModelAndPerformInference) {

// MicroErrorReporter overrides a mthod on ErrorReporter
// Create an ErrorReporter and point ot to micro_error_reporter
// A pointer doesnt hold a variable,(*) it holds a place in memory where the value exists
// User error_reporter for debugging

tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

// Map the model into a usable data structure. This doesn't involve any
// copying or parsing, it's a very lightweight operation.

// Get data array defined in simple_model, which returns a model pointer
// This gets assigned to variable model == simple_model
// The Model is a struct which is similar to a class in C++

const tflite::Model* model = ::tflite::GetModel(g_sine_model_data);
if (model->version() != TFLITE_SCHEMA_VERSION) { error_reporter->Report(
"Model provided is schema version %d not equal " "to supported version %d.\n",
model->version(), TFLITE_SCHEMA_VERSION); return 1;
}
