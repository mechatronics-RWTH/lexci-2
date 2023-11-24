
/*
 * Include Files
 *
 */
#if defined(MATLAB_MEX_FILE)
#include "tmwtypes.h"
#include "simstruc_types.h"
#else
#define SIMPLIFIED_RTWTYPES_COMPATIBILITY
#include "rtwtypes.h"
#undef SIMPLIFIED_RTWTYPES_COMPATIBILITY
#endif



/* %%%-SFUNWIZ_wrapper_includes_Changes_BEGIN --- EDIT HERE TO _END */
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

#include <memory>
#include <cstdint>
/* %%%-SFUNWIZ_wrapper_includes_Changes_END --- EDIT HERE TO _BEGIN */
#define u_width 3
#define u_1_width 1
#define u_2_width 1
#define u_3_width 65536
#define u_4_width 1
#define u_5_width 1
#define y_width 2

/*
 * Create external references here.  
 *
 */
/* %%%-SFUNWIZ_wrapper_externs_Changes_BEGIN --- EDIT HERE TO _END */
 
/* %%%-SFUNWIZ_wrapper_externs_Changes_END --- EDIT HERE TO _BEGIN */

/*
 * Output function
 *
 */
void PolicyNeuralNetwork_Outputs_wrapper(const real32_T *norm_observation,
			const uint32_T *norm_observation_size,
			const uint32_T *norm_action_dist_size,
			const uint8_T *static_nn_memory,
			const uint32_T *tensor_arena_size,
			const uint32_T *rl_algorithm,
			real32_T *norm_action_dist)
{
/* %%%-SFUNWIZ_wrapper_Outputs_Changes_BEGIN --- EDIT HERE TO _END */
// The static NN memory hasn't been overwritten.
if(static_nn_memory[0] == 0x00)
{
  for(unsigned int i = 0; i < *norm_action_dist_size; i++)
  {
    norm_action_dist[i] = 0;
  }
}
else
{
  std::unique_ptr<tflite::MicroErrorReporter> sptr_microErrorReporter;
  std::unique_ptr<tflite::AllOpsResolver> sptr_resolver;
  std::unique_ptr<uint8_t> sptr_tensorArena;
  std::unique_ptr<tflite::MicroInterpreter> sptr_interpreter;
  
  sptr_microErrorReporter = std::unique_ptr<tflite::MicroErrorReporter>(
      new tflite::MicroErrorReporter());
  sptr_resolver = std::unique_ptr<tflite::AllOpsResolver>(
      new tflite::AllOpsResolver());
  sptr_tensorArena = std::unique_ptr<uint8_t>(new uint8_t[*tensor_arena_size]);
  sptr_interpreter = std::unique_ptr<tflite::MicroInterpreter>(
      new tflite::MicroInterpreter(::tflite::GetModel(static_nn_memory),
      *sptr_resolver, sptr_tensorArena.get(), *tensor_arena_size,
      sptr_microErrorReporter.get()));
  sptr_interpreter->AllocateTensors();

  TfLiteTensor* ptr_input = sptr_interpreter->input(0);
  for(unsigned int i = 0; i < *norm_observation_size; i++)
  {
    ptr_input->data.f[i] = norm_observation[i];
  }
  sptr_interpreter->Invoke();
  
  TfLiteTensor* ptr_output;
  switch(*rl_algorithm)
  {
    case 1:  // PPO
      ptr_output = sptr_interpreter->output(0);
      break;
    case 2:  // DDPG
      ptr_output = sptr_interpreter->output(0);
      break;
    default:
      ptr_output = sptr_interpreter->output(0);
      break;
  }
  for(unsigned int i = 0; i < *norm_action_dist_size; i++)
  {
    norm_action_dist[i] = ptr_output->data.f[i];
  }
}
/* %%%-SFUNWIZ_wrapper_Outputs_Changes_END --- EDIT HERE TO _BEGIN */
}


