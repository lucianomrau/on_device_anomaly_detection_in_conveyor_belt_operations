#pragma once
#include <math.h>
#include <stdint.h>
#include <vector>
#include <array>

// DEFINES ///
//////////////////
#define N_VALUES 5  //historical values of the inputs (2 past, actual, 2 futures)
#define N_FEATURES 12    //number of features
#define N_INPUTS 3      //number of inputs variables
#define MEDIAN_FILTER_ORDER 3
#define MAX_SIZE_STATES 15 //maximo numero de cambios de estados normal



// VARIABLES ///
//////////////////
// TensorFlow Lite for Microcontroller global variables
const tflite::Model* tflu_model            = nullptr;
tflite::MicroInterpreter* tflu_interpreter = nullptr;
TfLiteTensor* tflu_i_tensor                = nullptr;
TfLiteTensor* tflu_o_tensor                = nullptr;
tflite::MicroErrorReporter tflu_error;

constexpr int tensor_arena_size = 2 * 1024;
byte tensor_arena[tensor_arena_size] __attribute__((aligned(16)));
float   tflu_i_scale      = 0.0f;
float   tflu_o_scale      = 0.0f;
int32_t tflu_i_zero_point = 0;
int32_t tflu_o_zero_point = 0;


uint8_t median_filter_buffer[MEDIAN_FILTER_ORDER]={0};
//unsigned int inference_time[REPEAT_EXP] = {};
//unsigned int inference_index = 0;
//unsigned int min_inference_time = 0xFFFFFFFF;
//unsigned int max_inference_time = 0;
//float mean_inference_time = 0;
//float std_inference_time = 0;
float v_features[N_FEATURES]={0};
float v_input[N_VALUES][N_INPUTS]={0};


//variables del detector de ciclo
bool flag_start_cycle=false;
bool flag_end_cycle=false;
uint8_t cycle_class;

//variables de entradas: high-pressure, low-pressure, speed
float f1, f2, f3;

typedef enum {A,B,C,D} state;
//typedef enum {Normal,Abnormal} cycle;

volatile state predic_state=A;
volatile state output_state=A;
volatile state output_state_m2=A;
volatile state output_state_m1=A;
std::vector<int> state_sequence;
const std::array<state, 3> pattern1 = {C,D,C};
const std::array<state, 2> pattern2 = {C,D};

//variables del clasificador de ciclos
bool b_condition1=false; //chequea los estados previous a la deteccion del ciclo
bool b_condition2=false;//chequea los estados dentro a la deteccion del ciclo
uint8_t c_after_cycle=0; //contador luego de finalizar la deteccion del ciclo para evaluar los estados posteriores

#ifdef FLOAT
  float v_features_quant[N_FEATURES]={0};
#elif defined INT8
  uint8_t v_features_quant[N_FEATURES]={0};
#endif

// MIN AND RANGE OF FEATURES FOR NORMALIZATION
const float feature_min[N_FEATURES]={0.0f, 0.0f,   0.0f,   0.0f,   0.0f, 0.0f, -1.34333333f, 0.0f,  0.0f,   0.0f, -1.194f,  -1.37f};
const float feature_range[N_FEATURES]={155.93f,41.73f,45.55f ,45.39333333f,155.35333333f,41.31333333f,122.16666667f,45.36f,154.672f,40.976f, 120.66f, 124.88f};

//#ifdef FLOAT
//  const float threshold_speed=2.5f/45.55f;
//#elif defined UINT32
//  const uint32_t threshold_speed=2.5f/45.55f;
//#elif defined UINT16
//  const uint16_t threshold_speed=2.5f/45.55f;
//#elif defined INT8
//  const uint8_t threshold_speed=2.5f/45.55f;
//#endif
const float threshold_speed=2.5f/45.55f;

// FUNCIONES

// Function to swap two elements in an array
void swap(uint8_t *x, uint8_t *y) {
    //float temp = *x;
    uint8_t temp = *x;
    *x = *y;
    *y = temp;
}

// Function to perform bubble sort on the array
void bubbleSort(uint8_t arr[], uint8_t n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(&arr[j], &arr[j + 1]);
            }
        }
    }
}    

uint8_t computeMedian(const uint8_t arr[]) {
    uint8_t temp_arr[3];
    temp_arr[0]=arr[0];
    temp_arr[1]=arr[1];
    temp_arr[2]=arr[2];
    bubbleSort(temp_arr, MEDIAN_FILTER_ORDER);
    return (uint8_t) temp_arr[MEDIAN_FILTER_ORDER / 2];
}


//funcion to compare the states inside the duty-cyle
template <size_t N>
bool compare_with_pattern(const std::array<state, N>& pattern) {
    if (state_sequence.size() != N) {  // Ensure sizes match
        return false;
    }

    for (size_t i = 0; i < state_sequence.size(); ++i) {
        if (state_sequence[i] != pattern[i]) {
            return false;
        }
    }
    return true;
}

void compute_features(const float v_input[N_VALUES][N_INPUTS],float *v_features)
{
  float high_pres_m2 = v_input[0][0];
  float high_pres_m1 = v_input[1][0];
  float high_pres = v_input[2][0];
  float high_pres_p1 = v_input[3][0];
  float high_pres_p2 = v_input[4][0];
  float low_pres_m2 = v_input[0][1];
  float low_pres_m1 = v_input[1][1];
  float low_pres = v_input[2][1];
  float low_pres_p1 = v_input[3][1];
  float low_pres_p2 = v_input[4][1];
  float speed_m2 = v_input[0][2];
  float speed_m1 = v_input[1][2];
  float speed = v_input[2][2];
  float speed_p1 = v_input[3][2];
  float speed_p2 = v_input[4][2];

  v_features[0]=high_pres;
  v_features[1]=low_pres;
  v_features[2]=speed;
  v_features[3]=(speed_m1+speed+speed_p1)/((float) 3);  //mean 3
  v_features[4]=(high_pres_m1+high_pres+high_pres_p1)/((float) 3);  //mean 3
  v_features[5]=(low_pres_m1+low_pres+low_pres_p1)/((float) 3);  //mean 3
  v_features[6]=v_features[4]-v_features[5];  //difference order 3
  v_features[7]=(speed_m2+speed_m1+speed+speed_p1+speed_p2)/((float) 5);  //mean 5
  v_features[8]=(high_pres_m2+high_pres_m1+high_pres+high_pres_p1+high_pres_p2)/((float) 5);  //mean 5
  v_features[9]=(low_pres_m2+low_pres_m1+low_pres+low_pres_p1+low_pres_p2)/((float) 5);  //mean 5
  v_features[10]=v_features[8]-v_features[9];
  v_features[11]=high_pres-low_pres;          
}

void normalize_features(float *v_features, const float *feature_min,const float *feature_range)
{
  for (uint8_t i=0;i<N_FEATURES;i++)
  {
    v_features[i]=(v_features[i]-feature_min[i])/feature_range[i];
  }    
}

#ifdef UINT32

uint32_t quantization(float value) {
      if (value < 0.0f) {
          return (uint32_t) 0;
      }
      else if (value>1.0f)
      {
        return (uint32_t) 4294967295;
      }
      else {
        return (uint32_t)(value * 4294967295.0f);  // 2^32 - 1
      }
}

void quantize_features(const float *v_features, uint32_t v_features_quant[])
{
  for (uint8_t i=0;i<N_FEATURES;i++)
  {
    v_features_quant[i]= quantization(v_features[i]);
  }    
}  

#elif UINT16
uint16_t quantization(float value) {
      if (value < 0.0f) {
          return 0;
      }
      else if (value>1.0f)
      {
        return 65535;
      }
      else {
        return (uint16_t)(value * 65535.0f);  // 2^16 - 1
      }
  }

void quantize_features(const float *v_features, uint16_t v_features_quant[])
{
  for (uint8_t i=0;i<N_FEATURES;i++)
  {
    v_features_quant[i]= quantization(v_features[i]);
  }    
}  

#elif defined INT8
uint8_t quantization(float value) {
      if (value < 0.0f) {
          return 0;
      }
      else if (value>1.0f)
      {
        return 255;
      }
      else {
        return (uint8_t)(value * 255.0f);  // 2^8 - 1
      }
  }

void quantize_features(const float *v_features, uint8_t v_features_quant[])
{
  for (uint8_t i=0;i<N_FEATURES;i++)
  {
    v_features_quant[i]= quantization(v_features[i]);
  }    
}  
#endif


void tflu_initialization()
{
  Serial.println("TFLu initialization - start");

  // Load the TFLITE model
  tflu_model = tflite::GetModel(model_tflite);
  if (tflu_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print(tflu_model->version());
    Serial.println("");
    Serial.print(TFLITE_SCHEMA_VERSION);
    Serial.println("");
    while (1);
  }

  tflite::AllOpsResolver tflu_ops_resolver;

  // Initialize the TFLu interpreter
  tflu_interpreter = new tflite::MicroInterpreter(tflu_model, tflu_ops_resolver, tensor_arena, tensor_arena_size, &tflu_error);
  //tflu_interpreter = new tflite::MicroInterpreter(tflu_model, tflu_ops_resolver, tensor_arena, tensor_arena_size);

  // Allocate TFLu internal memory
  tflu_interpreter->AllocateTensors();

  // Get the pointers for the input and output tensors
  tflu_i_tensor = tflu_interpreter->input(0);
  tflu_o_tensor = tflu_interpreter->output(0);

#ifdef INT8
  const auto* i_quantization = reinterpret_cast<TfLiteAffineQuantization*>(tflu_i_tensor->quantization.params);
  const auto* o_quantization = reinterpret_cast<TfLiteAffineQuantization*>(tflu_o_tensor->quantization.params);
  // Get the quantization parameters (per-tensor quantization)
  tflu_i_scale      = i_quantization->scale->data[0];
  tflu_i_zero_point = i_quantization->zero_point->data[0];
  tflu_o_scale      = o_quantization->scale->data[0];
  tflu_o_zero_point = o_quantization->zero_point->data[0];
#endif
  Serial.println("TFLu initialization - completed");
}

inline int8_t quantize(float x, float scale, float zero_point)
{
  return (x / scale) + zero_point;
}

inline float dequantize(int8_t x, float scale, float zero_point)
{
  return ((float)x - zero_point) * scale;
}
