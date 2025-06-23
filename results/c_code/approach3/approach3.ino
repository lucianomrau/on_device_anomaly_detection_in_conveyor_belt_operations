// choose the quantization type for the classifiers
#define INT8

/*
Uncomment for loading the input data of a duty cycle stored in flash memory. 
Cooment for reading the data from the serial port. The data from the serial port is sent by the Python script import_csv.py
*/
//#define DEBUG 


#include <vector>
#include <array>
#include <TensorFlowLite.h>
#include <tensorflow/lite/version.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
//#include <tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#ifdef FLOAT
  #include "mlp_float.h"
  #include "clf_duty_float.h"
#elif defined INT8
  #include "mlp_int8.h"
  #include "clf_duty_uint8.h"
#endif

#include "definitions.h"

void setup() {
  Serial1.begin(115200);  // Increased baud rate to match Python script
  while (!Serial1); 
  tflu_initialization(); 
  Serial1.print(tflu_interpreter->arena_used_bytes(),DEC);
#ifdef DEBUG
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);
  delay(1000);
#endif
}


#ifdef DEBUG
void loop() {
  
  #define N_SAMPLES 84
  #define N_REPEAT 100
  const float v_csv[N_SAMPLES][3]={
  {33.5,25.4,0},
  {33.5,25.4,0},
  {33.5,25.4,0},
  {33.5,25.3,0},
  {33.5,25.3,0},
  {33.5,25.3,0},
  {33.5,25.3,0},
  {43.7,32.3,0},
  {46.7,32.7,0},
  {46.7,32.5,0.32},
  {57,35.1,15.4},
  {62.4,37.4,40.8},
  {61.2,37,44.9},
  {60.4,36.6,45.2},
  {60,36.4,45.2},
  {59.6,36.2,45.2},
  {59.3,36,45.2},
  {59.2,35.8,45.3},
  {58.9,35.6,45.2},
  {58.7,35.4,45.2},
  {58.5,35.3,45.2},
  {58.4,35.1,45.3},
  {58,35,45.2},
  {57.7,34.8,45.3},
  {57.6,34.7,45.2},
  {57.5,34.5,45.2},
  {57.5,34.4,45.3},
  {57.5,34.4,45.3},
  {57.4,34.3,45.3},
  {57.2,34.3,45.3},
  {57,34.3,45.3},
  {57,34.4,45.2},
  {57.2,34.5,45.2},
  {57.4,34.6,45.2},
  {57.5,34.7,45.2},
  {57.5,34.7,45.2},
  {57.5,34.7,45.3},
  {57.5,34.6,45.2},
  {57.5,34.5,45.2},
  {57.3,34.4,45.2},
  {57.1,34.3,45.2},
  {56.9,34.2,45.2},
  {56.8,34.2,45.3},
  {57.1,34.2,45.2},
  {58.9,34.3,45.2},
  {60.6,34.4,45.2},
  {61.7,34.5,45.2},
  {62.3,34.5,45.2},
  {67.1,34.6,45.2},
  {74.1,34.6,45.2},
  {87.1,34.5,45},
  {116,34.2,44.7},
  {135,34.1,44.5},
  {136,34,44.4},
  {137,33.9,44.5},
  {136,33.9,44.4},
  {137,33.9,44.4},
  {137,33.9,44.4},
  {138,33.9,44.4},
  {138,34,44.4},
  {141,34,44.4},
  {145,34,44.3},
  {146,34,44.3},
  {146,34,44.3},
  {146,34,44.3},
  {147,33.9,44.3},
  {145,33.9,44.3},
  {137,34,44.5},
  {128,34,44.6},
  {101,34.3,45},
  {65.9,34.4,45.3},
  {56.7,34.3,45.3},
  {56.7,34.4,45.2},
  {56.9,34.4,45.2},
  {56.8,34.5,45.2},
  {56.9,34.5,45.2},
  {53.4,33.8,41.4},
  {43.3,32.3,6.67},
  {43.9,33,0},
  {41.8,33.3,0},
  {41.8,33.4,0},
  {41.9,33.4,0},
  {41.8,33.4,0},
  {38.7,29.7,0}
};
delay(1000);
digitalWrite(LED_BUILTIN, LOW);
for(int l=0;l<N_REPEAT;l++){
 for(int k=0;k<N_SAMPLES;k++){
    f1 = v_csv[k][0];
    f2 = v_csv[k][1];
    f3 = v_csv[k][2];



    //INSERT VALUE IN THE HISTORY BUFFERS
    for(uint8_t i=0;i<N_VALUES-1;i++){
      v_input[i][0]=v_input[i+1][0];
      v_input[i][1]=v_input[i+1][1];
      v_input[i][2]=v_input[i+1][2];
    }
    v_input[N_VALUES-1][0]=f1;
    v_input[N_VALUES-1][1]=f2;
    v_input[N_VALUES-1][2]=f3;
        
    
    //EXTRACT FEATURES
    compute_features(v_input,v_features);

    //NORMALIZATION
    normalize_features(v_features,feature_min,feature_range);
    
    #ifdef FLOAT
        for(int i = 0; i < N_FEATURES; i++) {
          tflu_i_tensor->data.f[i] = v_features[i];
        }
  
        //PREDICTION
        TfLiteStatus invoke_status = tflu_interpreter->Invoke();
          if (invoke_status != kTfLiteOk) {
            Serial1.println("Error invoking the TFLu interpreter");
          return;
        }
    
        predic_state = (state) 0;
        float  pb_max = 0;  //max probability output


        for (uint8_t ix = 0; ix < 4; ix++) {
          float out_f = tflu_o_tensor->data.f[ix];
          if(tflu_o_tensor->data.f[ix] > pb_max) {
            predic_state = (state) ix;
            pb_max = tflu_o_tensor->data.f[ix];
          }
        }

    #else
        //QUANTIZATION 
        for(int i = 0; i < N_FEATURES; i++) {
          int8_t feature_q = quantize(v_features[i], tflu_i_scale, tflu_i_zero_point);
          tflu_i_tensor->data.int8[i] = feature_q;
        }

        //PREDICTION
        TfLiteStatus invoke_status = tflu_interpreter->Invoke();
          if (invoke_status != kTfLiteOk) {
            Serial1.println("Error invoking the TFLu interpreter");
          return;
        }

        predic_state = (state) 0;
        float  pb_max = 0;  //max probability output


        for (uint8_t ix = 0; ix < 4; ix++) {
          int8_t out_int8 = tflu_o_tensor->data.int8[ix];
          float out_f = dequantize(out_int8, tflu_o_scale, tflu_o_zero_point);
          if(out_f > pb_max) {
            predic_state = (state) ix;
            pb_max = out_f;
          }
        }
      
    #endif


    //COMPUTE MEDIAN FILTER
    //This filter introduces me a delay of a sample (Z-1), therefore I will operate the cycle detector, a delayed sample
    median_filter_buffer[0]=median_filter_buffer[1];
    median_filter_buffer[1]=median_filter_buffer[2];
    median_filter_buffer[2]=predic_state;
    output_state = (state) computeMedian(median_filter_buffer);
    
    //CYCLE DETECTION
    if ((v_features[3] >= threshold_speed) & (!flag_start_cycle) & (!flag_end_cycle)){
    //step1: detect the start of the cycle
      flag_start_cycle=true;
      state_sequence.push_back(output_state_m2);
      if (output_state_m2!=output_state_m1){
        state_sequence.push_back(output_state_m1);
      }
      if (output_state_m1!=output_state){
        state_sequence.push_back(output_state);
      }
    }
      
    else if((v_features[3] >= threshold_speed) && flag_start_cycle && !flag_end_cycle){
      //step2: iterate inside the cycle and save the state changing
      if (output_state!=output_state_m1 && (state_sequence.size()<MAX_SIZE_STATES)){ // si hay cambio de estados
        state_sequence.push_back(output_state);
      }
    }
    
    else if(v_features[3] < threshold_speed && flag_start_cycle && !flag_end_cycle){
      //step3: find the end of the cycle
      flag_end_cycle=true;
      c_after_cycle=0;
      if (output_state!=output_state_m1 && (state_sequence.size()<MAX_SIZE_STATES)){ // si hay cambio de estados
        state_sequence.push_back(output_state);
      }
    }
    
    else if(flag_start_cycle && flag_end_cycle && c_after_cycle!=2){
      //step4: count the state after end of the cycle
      c_after_cycle++;
      if (output_state!=output_state_m1 && (state_sequence.size()<MAX_SIZE_STATES)){ // si hay cambio de estados
        state_sequence.push_back(output_state);
      }
    }
    else if(flag_start_cycle && flag_end_cycle && c_after_cycle==2){
      //step5: make the classification
      //PADDING AND CLASSIFICATION
      while(state_sequence.size()<MAX_SIZE_STATES){
        state_sequence.push_back(A);
      }
      #ifdef FLOAT
        float tmp_array[MAX_SIZE_STATES];
        for (uint8_t i=0;i<MAX_SIZE_STATES;i++){
          tmp_array[i]= (float) state_sequence[i];
        }
      #else
        uint8_t tmp_array[MAX_SIZE_STATES];
        for (uint8_t i=0;i<MAX_SIZE_STATES;i++){
          tmp_array[i]= (float) state_sequence[i];
        }
      #endif
      cycle_class = (uint8_t) clf_duty_predict(tmp_array,MAX_SIZE_STATES);
      if (cycle_class){
        Serial1.println("Normal");
      }
      else{
        Serial1.println("Abnormal");
      }

      state_sequence.clear();
      flag_start_cycle=false;
      flag_end_cycle=false;

    }


  //I keep the last three positions in case I detect a cycle
  output_state_m2 = output_state_m1;
  output_state_m1 = output_state;
}
}
digitalWrite(LED_BUILTIN, HIGH);
while(true);
}

#else
void loop() {
    

  //read 3 inputs values from serie port
  if (Serial1.available() >= 12) {  // Expect 12 bytes (3 floats * 4 bytes each)
    byte buffer[12];
    Serial1.readBytes(buffer, 12);
    // Convert bytes to floats (Arduino is already little-endian)
    memcpy(&f1, buffer, 4);
    memcpy(&f2, buffer + 4, 4);
    memcpy(&f3, buffer + 8, 4);



    //INSERT VALUE IN THE HISTORY BUFFERS
    for(uint8_t i=0;i<N_VALUES-1;i++){
      v_input[i][0]=v_input[i+1][0];
      v_input[i][1]=v_input[i+1][1];
      v_input[i][2]=v_input[i+1][2];
    }
    v_input[N_VALUES-1][0]=f1;
    v_input[N_VALUES-1][1]=f2;
    v_input[N_VALUES-1][2]=f3;
        
    
    //EXTRACT FEATURES
    compute_features(v_input,v_features);

    //NORMALIZATION
    normalize_features(v_features,feature_min,feature_range);
    
    #ifdef FLOAT
        for(int i = 0; i < N_FEATURES; i++) {
          tflu_i_tensor->data.f[i] = v_features[i];
        }
  
  
        //PREDICTION
        TfLiteStatus invoke_status = tflu_interpreter->Invoke();
//          if (invoke_status != kTfLiteOk) {
//            Serial1.println("Error invoking the TFLu interpreter");
//          return;
//        }
    
    
        predic_state = (state) 0;
        float  pb_max = 0;  //max probability output


        for (uint8_t ix = 0; ix < 4; ix++) {
          float out_f = tflu_o_tensor->data.f[ix];
          if(tflu_o_tensor->data.f[ix] > pb_max) {
            predic_state = (state) ix;
            pb_max = tflu_o_tensor->data.f[ix];
          }
        }
        
    #else
        //QUANTIZATION 
        for(int i = 0; i < N_FEATURES; i++) {
          int8_t feature_q = quantize(v_features[i], tflu_i_scale, tflu_i_zero_point);
          tflu_i_tensor->data.int8[i] = feature_q;
        }

        //PREDICTION
        TfLiteStatus invoke_status = tflu_interpreter->Invoke();
          if (invoke_status != kTfLiteOk) {
            Serial1.println("Error invoking the TFLu interpreter");
          return;
        }

        predic_state = (state) 0;
        float  pb_max = 0;  //max probability output


        for (uint8_t ix = 0; ix < 4; ix++) {
          int8_t out_int8 = tflu_o_tensor->data.int8[ix];
          float out_f = dequantize(out_int8, tflu_o_scale, tflu_o_zero_point);
          if(out_f > pb_max) {
            predic_state = (state) ix;
            pb_max = out_f;
          }
        }
      
    #endif



    //COMPUTE MEDIAN FILTER
//This filter introduces me a delay of a sample (Z-1), therefore I will operate the cycle detector, a delayed sample
    median_filter_buffer[0]=median_filter_buffer[1];
    median_filter_buffer[1]=median_filter_buffer[2];
    median_filter_buffer[2]=predic_state;
    output_state = (state) computeMedian(median_filter_buffer);

    //CYCLE DETECTION
    if ((v_features[3] >= threshold_speed) & (!flag_start_cycle) & (!flag_end_cycle)){
    //step1: detect the start of the cycle
      flag_start_cycle=true;
      state_sequence.push_back(output_state_m2);
      if (output_state_m2!=output_state_m1){
        state_sequence.push_back(output_state_m1);
      }
      if (output_state_m1!=output_state){
        state_sequence.push_back(output_state);
      }
    }
      
    else if((v_features[3] >= threshold_speed) && flag_start_cycle && !flag_end_cycle){
      //step2: iterate inside the cycle and save the state changing
      if (output_state!=output_state_m1 && (state_sequence.size()<MAX_SIZE_STATES)){ // si hay cambio de estados
        state_sequence.push_back(output_state);
      }
    }
    
    else if(v_features[3] < threshold_speed && flag_start_cycle && !flag_end_cycle){
      //step3: find the end of the cycle
      flag_end_cycle=true;
      c_after_cycle=0;
      if (output_state!=output_state_m1 && (state_sequence.size()<MAX_SIZE_STATES)){ // si hay cambio de estados
        state_sequence.push_back(output_state);
      }
    }
    
    else if(flag_start_cycle && flag_end_cycle && c_after_cycle!=2){
      //step4: count the state after end of the cycle
      c_after_cycle++;
      if (output_state!=output_state_m1 && (state_sequence.size()<MAX_SIZE_STATES)){ // si hay cambio de estados
        state_sequence.push_back(output_state);
      }
    }
    else if(flag_start_cycle && flag_end_cycle && c_after_cycle==2){
      //step5: make the classification
      //PADDING AND CLASSIFICATION
      while(state_sequence.size()<MAX_SIZE_STATES){
        state_sequence.push_back(A);
      }
      #ifdef FLOAT
        float tmp_array[MAX_SIZE_STATES];
        for (uint8_t i=0;i<MAX_SIZE_STATES;i++){
          tmp_array[i]= (float) state_sequence[i];
        }
      #else
        uint8_t tmp_array[MAX_SIZE_STATES];
        for (uint8_t i=0;i<MAX_SIZE_STATES;i++){
          tmp_array[i]= (float) state_sequence[i];
        }
      #endif
      cycle_class = (uint8_t) clf_duty_predict(tmp_array,MAX_SIZE_STATES);
      if (cycle_class){
        Serial1.println("Normal");
      }
      else{
        Serial1.println("Abnormal");
      }

      state_sequence.clear();
      flag_start_cycle=false;
      flag_end_cycle=false;

    }


  //I keep the last three positions in case I detect a cycle
  output_state_m2 = output_state_m1;
  output_state_m1 = output_state;
}
}
#endif
