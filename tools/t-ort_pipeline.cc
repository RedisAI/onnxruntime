// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.

// Compile
// g++ -std=c++11 -o t-ort_pipeline t-ort_pipeline.cc -I /bert_ort/pranav/onnxruntime/include/onnxruntime/core/session -I /bert_ort/pranav/onnxruntime/include/onnxruntime/core/providers/cuda/ -lonnxruntime -lpthread -ltbb -L build/Linux/Debug/ -Wl,-rpath,/bert_ort/pranav/onnxruntime/build/Linux/Debug/

#include <assert.h>
#include <vector>
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include "/bert_ort/pranav/onnxruntime/include/onnxruntime/core/providers/cuda/cuda_provider_factory.h"
#include <iostream>
#include <thread>
#include "tbb/pipeline.h"
#include <unordered_map>
#include <memory>

/*
* This is just a prototype to demonstrate the usage of Intel TBB's parallel_pipeline to implement
* pipeline parallelism. See main() for usage.
* It runs 2 models on 2 separate GPU devices in a pipeline.
* The output of first model is allocated on GPU-0 and fed to the next model running on GPU-1.
* The cross-device copy (from GPU-0 to GPU-1) is done by ORT as part of Run().
*/

const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

// helper function to check for status
void CheckStatus(OrtStatus* status) {
  if (status != NULL) {
    const char* msg = g_ort->GetErrorMessage(status);
    fprintf(stderr, "%s\n", msg);
    g_ort->ReleaseStatus(status);
    exit(1);
  }
}

struct PipelineSession {
  std::shared_ptr<OrtStatus> Run(const std::vector<std::string>& input_names,
                                 std::vector<Ort::Value>& input_values,  // TODO should be const
                                 const std::vector<std::string>& output_names,
                                 std::vector<Ort::Value>& output_values) {
    using namespace tbb;
    int i = 0;
    int batch_size = 1;

    auto input_stage_fn = [&](flow_control& fc) {
      if (i++ < batch_size) {
        std::vector<Ort::Value> dummy;
        return std::shared_ptr<Token>(new Token{std::string(), input_names, std::move(input_values)});
      } else {
        fc.stop();
        return std::shared_ptr<Token>(nullptr);
      }
    };
    auto input_stage_filter = make_filter<void, std::shared_ptr<Token>>(filter::serial, input_stage_fn);

    auto output_stage_fn = [&](std::shared_ptr<Token> token) {
      for (auto& elem : token->input_values) {
        output_values.push_back(std::move(elem));
      }
    };
    auto output_stage_filter = make_filter<std::shared_ptr<Token>, void>(filter::serial, output_stage_fn);

    auto model_exec_fn = [](int stage_id, Token& token, const std::string& model_name, PipelineSession& psess) {
      std::cout << "Executing model_name: " << model_name << "\n";

      auto& model_config = psess.ens.models[psess.model_configs.at(model_name)];
      auto& run_config = psess.run_configs.at(model_name);

      auto& ort_sess = run_config.session;
      auto* device_allocator = run_config.device_allocator.get();

      // TODO get input and output shape
      std::vector<const char*> input_names;
      input_names.reserve(token.input_names.size());
      for (const auto& elem : token.input_names) {
        input_names.push_back(elem.c_str());
      }

      std::vector<const char*> output_names;
      for (const auto& elem : model_config.output_names) {
        output_names.push_back(elem.c_str());
      }

      auto token_ptr = std::make_shared<Token>();
      if (stage_id == psess.ens.models.size() - 1) {
        std::vector<Ort::Value> output_values = ort_sess.Run(Ort::RunOptions{nullptr}, input_names.data(),
                                                             token.input_values.data(), token.input_values.size(),
                                                             output_names.data(), output_names.size());
        // now populate token
        for (int i = 0; i < output_names.size(); ++i) {
          // get input name from the map
          token_ptr->input_names.push_back(model_config.output_input_map[output_names[i]]);
          token_ptr->input_values.push_back(std::move(output_values[i]));
        }
      } else {
        std::vector<int64_t> ouptut_node_dims{3, 2};  // TODO remove this
        auto output_tensor = Ort::Value::CreateTensor<float>(device_allocator, ouptut_node_dims.data(),
                                                             ouptut_node_dims.size());
        std::vector<Ort::Value> output_values;
        output_values.push_back(std::move(output_tensor));

        ort_sess.Run(Ort::RunOptions{nullptr}, input_names.data(), token.input_values.data(), token.input_values.size(),
                     output_names.data(), output_values.data(), output_names.size());

        // now populate token
        for (int i = 0; i < output_names.size(); ++i) {
          // get input name from the map
          token_ptr->input_names.push_back(model_config.output_input_map[output_names[i]]);
          token_ptr->input_values.push_back(std::move(output_values[i]));
        }
      }

      return token_ptr;
    };

    // create filter based on first model
    auto model_exec_filter_chain =
        make_filter<std::shared_ptr<Token>,
                    std::shared_ptr<Token>>(filter::parallel,
                                            [this, &model_exec_fn, &model_name = ens.models[0].model_name](std::shared_ptr<Token> token_ptr) {
                                              return model_exec_fn(0, *token_ptr, model_name, *this);
                                            });

    // join filters from other models
    for (int i = 1; i < ens.models.size(); ++i) {
      model_exec_filter_chain = model_exec_filter_chain &
                                make_filter<std::shared_ptr<Token>,
                                            std::shared_ptr<Token>>(filter::parallel,
                                                                    [this, i, &model_exec_fn, &model_name = ens.models[i].model_name](std::shared_ptr<Token> token_ptr) {
                                                                      return model_exec_fn(i, *token_ptr, model_name, *this);
                                                                    });
    }

    // create and run the pipeline
    parallel_pipeline(degree_of_parallelism,
                      input_stage_filter & model_exec_filter_chain & output_stage_filter);
  }

  PipelineSession(Ort::Env& env /*json config file path*/) {
    std::string model_name1 = "mul_1";
    std::string model_name2 = "mul_2";
    std::string input_name = "X";
    std::string output_name = "Y";
    std::string model_path = "/bert_ort/pranav/onnxruntime/onnxruntime/test/testdata/mul_1.onnx";
    ModelConfig m1{model_name1,
                   model_path,
                   {input_name},
                   {output_name},
                   {{output_name, input_name}},
                   0};

    ModelConfig m2{model_name2,
                   model_path,
                   {input_name},
                   {output_name},
                   {{output_name, input_name}},
                   1};

    ens.models.push_back(std::move(m1));
    model_configs["mul_1"] = ens.models.size() - 1;

    ens.models.push_back(std::move(m2));
    model_configs["mul_2"] = ens.models.size() - 1;

    for (const auto& mcfg : ens.models) {
      // create session
      Ort::SessionOptions session_options;
      CheckStatus(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, mcfg.device_id));
      Ort::Session session(env, mcfg.model_file_path.c_str(), session_options);

      // create device allocator
      OrtAllocator* device_allocator = nullptr;
      Ort::MemoryInfo mem_info{
          "Cuda",
          OrtArenaAllocator,
          mcfg.device_id,
          OrtMemTypeDefault,
      };
      CheckStatus(g_ort->CreateAllocator(session, mem_info, &device_allocator));
      std::unique_ptr<OrtAllocator, decltype(g_ort->ReleaseAllocator)> u_device_allocator(device_allocator, g_ort->ReleaseAllocator);
      RunConfig rcfg{std::move(session), std::move(u_device_allocator)};
      run_configs.emplace(mcfg.model_name, std::move(rcfg));
    }
  }

  // data members
  using OutputInputMap = std::unordered_map<std::string, std::string>;

  struct ModelConfig {
    std::string model_name;
    std::string model_file_path;
    std::vector<std::string> input_names;   // TODO can be obtained from model as well?
    std::vector<std::string> output_names;  // TODO can be obtained from model as well?
    // TODO we need shape info as well for inputs and outputs?
    OutputInputMap output_input_map;  // maps output of this step to input of the next step
    int device_id;
    // TODO assume GPU for now; should record which device a user wants to run the model on
  };

  struct Ensemble {
    std::vector<ModelConfig> models;
  };

  struct Token {
    std::string error_msg;
    std::vector<std::string> input_names;
    std::vector<Ort::Value> input_values;
  };

  struct RunConfig {
    using OrtAllocatorUptr = std::unique_ptr<OrtAllocator, decltype(g_ort->ReleaseAllocator)>;
    Ort::Session session;
    OrtAllocatorUptr device_allocator;
  };

  int degree_of_parallelism = 10;  // TODO
  std::unordered_map<std::string, RunConfig> run_configs;
  std::unordered_map<std::string, int> model_configs;
  Ensemble ens;
};

int main(int argc, char* argv[]) {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

  // setup the pipeline session
  PipelineSession pipeline_session(env /*, "/tmp/ensemble.json"*/);

  // prepare inputs
  size_t input_tensor_size = 3 * 2;
  std::vector<int64_t> input_node_dims{3, 2};
  std::vector<std::string> input_node_names{"X"};
  std::vector<float> input_tensor_values(input_tensor_size);
  std::vector<std::string> output_node_names = {"Y"};

  int c = 1;
  for (unsigned int i = 0; i < input_tensor_size; i++) {
    input_tensor_values[i] = (float)c++;
  }
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size,
                                                      input_node_dims.data(), input_node_dims.size());
  assert(input_tensor.IsTensor());
  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(input_tensor));

  // Run the pipeline
  std::vector<Ort::Value> output_values;
  pipeline_session.Run(input_node_names, ort_inputs, output_node_names, output_values);

  // print output
  auto* data_ptr = output_values[0].GetTensorData<float>();
  std::cout << "Printing output " << std::endl;
  for (int i = 0; i < 3 * 2; ++i) {
    std::cout << data_ptr[i] << " ";
  }
  std::cout << std::endl;

  printf("Done!\n");
  return 0;
}
