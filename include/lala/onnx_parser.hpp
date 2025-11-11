// Copyright 2025 Yi-Nung Tsao

#ifndef LALA_PARSING_ONNX_PARSER_HPP
#define LALA_PARSING_ONNX_PARSER_HPP

#include <cassert>
#include <cstdlib>
#include <fstream>
#include <google/protobuf/message_lite.h>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

#include "battery/shared_ptr.hpp"
#include "battery/vector.hpp"
#include "lala/logic/ast.hpp"
#include "onnx-1.15.0/onnx.proto3.pb.h"
#include "solver_output.hpp"

namespace lala {

using tensor1d = battery::vector<float>;
using tensor2d = battery::vector<tensor1d>;
using tensor3d = battery::vector<tensor2d>;
using tensor4d = battery::vector<tensor3d>;

enum class LayerType {
  Sub,
  Div,
  Constant,
  MatMul,
  Add,
  Gemm,
  Conv,
  Flatten,
  Relu,
  Unknown
};

struct Layer {
  size_t size;
  size_t id;
  LayerType type;
  battery::vector<std::string> neurons;

  tensor1d sub_values;
  tensor1d div_values;

  tensor1d biases;
  tensor2d weights;
  tensor4d conv_weights;

  size_t input_dim;
  size_t output_dim;

  size_t input_height;
  size_t input_width;

  size_t kernel_height;
  size_t kernel_width;

  size_t input_channels;
  size_t output_channels;

  size_t conv_input_height;
  size_t conv_input_width;
  size_t conv_output_height;
  size_t conv_output_width;

  size_t dilations;
  size_t group;
  size_t pads;
  size_t strides;
};
} // namespace lala

namespace lala {

namespace impl {

template <class Allocator> class OnnxParser {
  using allocator_type = Allocator;
  using F = TFormula<allocator_type>;
  using So = Sort<allocator_type>;
  using FSeq = typename F::Sequence;

  bool error;  // If an error was found during parsing.
  bool silent; // If we do not want to output error messages.

public:
  OnnxParser() : error(false), silent(false) {}

  battery::shared_ptr<F, allocator_type>
  parser(const std::string &onnx_model_directory) {
    onnx::ModelProto model;
    std::ifstream input(onnx_model_directory, std::ios::in | std::ios::binary);

    if (!input) {
      std::cerr << "Failed to open: " + onnx_model_directory << std::endl;
      return nullptr;
    }

    if (!model.ParseFromIstream(&input)) {
      std::cerr << "Failed to parse onnx file." << std::endl;
      return nullptr;
    }

    const onnx::GraphProto &graph = model.graph();
    std::cout << "onnx model name: " << graph.name() << std::endl;
    std::cout << "number of nodes: " << graph.node_size() << std::endl;
    std::cout << "number of initializers (weights/biases): "
              << graph.initializer_size() << std::endl;

    // Create a map from tensor name to TensorProto for fast lookup
    std::unordered_map<std::string, onnx::TensorProto> tensor_map;
    for (const auto &initializer : graph.initializer()) {
      tensor_map[initializer.name()] = initializer;
      std::cout << initializer.name() << std::endl;
    }

    // Build a map from tensor name -> node that produces it
    // This is for linking sub/div node and its input nodes.
    std::unordered_map<std::string, const onnx::NodeProto *> producer_map;
    for (const auto &node : graph.node()) {
      if (node.output_size() > 0) {
        producer_map[node.output(0)] = &node;
      }
    }

    // Shape of input.
    const onnx::ValueInfoProto &graph_input = graph.input(0);
    const auto &input_shape =
        graph_input.type()
            .tensor_type()
            .shape(); // <batch_size, num_input_channels, input_H, input_W>;
    size_t batch_size = input_shape.dim(0).dim_value();
    size_t input_channels = input_shape.dim(1).dim_value();
    size_t input_height = input_shape.dim(2).dim_value();
    size_t input_width = input_shape.dim(3).dim_value();
    size_t input_dimensions = batch_size * input_channels * input_height *
                              input_width; // number of input neurons.

    // Create an input layer.
    battery::vector<Layer> layers;
    // NOTE: we do not need to make input_layer.
    // Because it would be made in VnnlibParser.
    // We create input_layer, just for later layers to know the input
    // dimensions.
    Layer input_layer;
    input_layer.id = layers.size();
    input_layer.size = input_dimensions;
    input_layer.input_height = input_height;
    input_layer.input_width = input_width;
    for (size_t i = 0; i < input_layer.size; ++i) {
      input_layer.neurons.push_back("X_" + std::to_string(i));
    }
    layers.push_back(input_layer);

    // TODO: Iterate over nodes in the network graph
    for (const auto &node : graph.node()) {
      Layer layer;
      layer.id = layers.size();
      layer.type = setLayerType(node);
      if (layer.type == LayerType::Unknown) {
        std::cerr << "Unknown layer type." << std::endl;
        return;
      }

      for (const auto &input_name : node.input()) {
        // If the input is a weight tensor
        if (tensor_map.find(input_name) != tensor_map.end()) {
          // In this case, node.op_type() could be Gemm, MatMul, Add, Conv
          const auto &tensor = tensor_map[input_name];
          if (tensor.dims().size() == 1) {
            // bias 1d tensor
            layer.biases = extract1DTensorData(tensor);
            if (node.op_type() == "Conv") {
              tensor1d expanded_biases(layer.size, 0.0f);
              for (size_t i = 0; i < layer.biases.size(); ++i) {
                for (size_t j = 0;
                     j < layer.conv_output_height * layer.conv_output_width;
                     ++j) {
                  expanded_biases.push_back(layer.biases[i]);
                }
              }
              layer.biases = expanded_biases;
            }
          } else if (tensor.dims().size() == 2) {
            // weight 2d tensor
            layer.weights = extract2DTensorData(tensor);

            // <output_dimensions, input_dimensions>
            layer.size = layer.weights[0].size();
          } else if (tensor.dims().size() == 4) {
            // Extract the attributes in convolutional layer.
            for (const auto &attr : node.attribute()) {
              if (attr.ints_size() > 0) {
                const std::string &attr_name = attr.name();
                if (attr_name == "dilations") {
                  layer.dilations = attr.ints()[0];
                } else if (attr_name == "group") {
                  layer.group = attr.ints()[0];
                } else if (attr_name == "kernel_shape") {
                  layer.kernel_height = attr.ints()[0];
                  layer.kernel_width = attr.ints()[0];
                } else if (attr_name == "pads") {
                  layer.pads = attr.ints()[0];
                } else if (attr_name == "strides") {
                  layer.strides = attr.ints()[0];
                }
              }
            }

            // conv weight 4d tensor
            layer.conv_weights = extract4DTensorData(tensor);
            layer.weights = convert4Dto2DTensor(
                layer.conv_weights, layer.conv_input_height,
                layer.conv_input_width, layer.strides, layer.pads);

            // update convolution input & output parameters
            layer.size = layer.weights.size();
            layer.output_dim = layer.conv_weights.size();
            layer.input_dim = layer.conv_weights[0].size();
            layer.kernel_height = layer.conv_weights[0][0].size();
            layer.kernel_width = layer.conv_weights[0][0][0].size();
            layer.input_channels = layer.input_dim;
            layer.output_channels = layer.output_dim;
            layer.conv_input_height = input_height;
            layer.conv_input_width = input_width;
            layer.conv_output_height = (layer.conv_input_height +
                                        2 * layer.pads - layer.kernel_height) /
                                           layer.strides +
                                       1;
            layer.conv_output_width =
                (layer.conv_input_width + 2 * layer.pads - layer.kernel_width) /
                    layer.strides +
                1;
            input_height =
                std::floor((input_height + 2 * layer.pads -
                            layer.dilations * (layer.kernel_height - 1) - 1) /
                               layer.strides +
                           1);
            input_width =
                std::floor((input_width + 2 * layer.pads -
                            layer.dilations * (layer.kernel_width - 1) - 1) /
                               layer.strides +
                           1);
          }
        } else if (node.op_type() == "Sub" || node.op_type() == "Div") {
          // finding the constant in sub/div node
          if (producer_map.find(input_name) != producer_map.end()) {
            const onnx::NodeProto *producer = producer_map[input_name];
            for (const auto &attr : producer->attribute()) {
              if (attr.has_t()) {
                // attribute constains a tensor
                const onnx::TensorProto &tensor = attr.t();
                if (tensor.data_type() == onnx::TensorProto::FLOAT) {
                  const std::string &raw = tensor.raw_data();
                  const float *data =
                      reinterpret_cast<const float *>(raw.data());
                  size_t numel = raw.size() / sizeof(float);
                  for (size_t i = 0; i < numel; ++i) {
                    if (layer.type == LayerType::Sub) {
                      layer.sub_values.push_back(data[i]);
                    } else if (layer.type == LayerType::Div) {
                      layer.div_values.push_back(data[i]);
                    }
                  }
                }
              } else if (attr.floats_size() > 0) {
                for (auto f : attr.floats()) {
                  if (layer.type == LayerType::Sub) {
                    layer.sub_values.push_back(f);
                  } else if (layer.type == LayerType::Div) {
                    layer.div_values.push_back(f);
                  }
                }
              }
            }
            layer.size = input_dimensions;
            layer.input_height = input_height;
            layer.input_width = input_width;
          }
        } else if (node.op_type() == "Constant") {
          continue;
        } else if (node.op_type() == "Flatten") {
          layer.size = layers[layers.size() - 1].size;
        } else if (node.op_type() == "Relu") {
          layer.size = layers[layers.size() - 1].size;
        }
      }

      // build constraint for this node/layer
      make_node(layers[layer.id - 1], layer);

      // The outputs of the node
      for (const auto &output_name : node.output()) {
        std::cout << "  Output: " << output_name << std::endl;
      }

      // store into list of layers
      layers.push_back(layer);
    }

    google::protobuf::ShutdownProtobufLibrary();

    return;
  }

private:
  static F f(const std::any &any) { return std::any_cast<F>(any); }

  LayerType setLayerType(const onnx::NodeProto &node) {
    if (node.op_type() == "Sub") {
      return LayerType::Sub;
    } else if (node.op_type() == "Div") {
      return LayerType::Div;
    } else if (node.op_type() == "Constant") {
      return LayerType::Constant;
    } else if (node.op_type() == "Flatten") {
      return LayerType::Flatten;
    } else if (node.op_type() == "MatMul") {
      return LayerType::MatMul;
    } else if (node.op_type() == "Add") {
      return LayerType::Add;
    } else if (node.op_type() == "Gemm") {
      return LayerType::Gemm;
    } else if (node.op_type() == "Conv") {
      return LayerType::Conv;
    } else if (node.op_type() == "Relu") {
      return LayerType::Relu;
    } else {
      return LayerType::Unknown;
    }
  }

  void make_node(const Layer &from_layer, const Layer &to_layer) {
    switch (to_layer.type) {
    case LayerType::Sub:
      make_sub_node(from_layer, to_layer);
      break;
    case LayerType::Div:
      make_sub_node(from_layer, to_layer);
      break;
    case LayerType::Constant:
      make_constant_node(from_layer, to_layer);
      break;
    case LayerType::Flatten:
      make_flatten_node(from_layer, to_layer);
      break;
    case LayerType::MatMul:
      make_matmul_node(from_layer, to_layer);
      break;
    case LayerType::Add:
      make_add_node(from_layer, to_layer);
      break;
    case LayerType::Gemm:
      make_gemm_node(from_layer, to_layer);
      break;
    case LayerType::Conv:
      make_conv_node(from_layer, to_layer);
      break;
    case LayerType::Relu:
      make_relu_node(from_layer, to_layer);
      break;
    default:
      break;
    }

    return;
  }

  void make_sub_node(const Layer &from_layer, const Layer &to_layer) {
    for (size_t i = 0; i < to_layer.size; ++i) {
      int dim = i / (to_layer.input_height * to_layer.input_width);

      // create variable
      auto ty = So(So::Real);
      auto var = F::make_exists(UNTYPED, LVar<allocator_type>(), ty);

      // x' = x - sub_value;
      F::make_binary(f(var), EQ,
                     F::make_binary(f(from_layer.neurons[i]), SUB,
                                    f(to_layer.sub_values[dim])));
    }

    return;
  }

  void make_div_node(const Layer &from_layer, const Layer &to_layer) {
    for (size_t i = 0; i < to_layer.size; ++i) {
      int dim = i / (to_layer.input_height * to_layer.input_width);

      // create variable
      auto ty = So(So::Real);
      auto var = F::make_exists(UNTYPED, LVar<allocator_type>(), ty);

      // x' = x / div_value;
      F::make_binary(f(var), EQ,
                     F::make_binary(f(from_layer.neurons[i]), DIV,
                                    f(to_layer.sub_values[dim])));
    }

    return;
  }

  void make_flatten_node(const Layer &from_layer, const Layer &to_layer) {
    // we have flatten input as a vector.
    // we do not need to do additional transformation here.
    return;
  }

  void make_add_node(const Layer &from_layer, const Layer &to_layer) {
    for (size_t i = 0; i < to_layer.size; ++i) {
      // create variable
      auto ty = So(So::Real);
      auto var = F::make_exists(UNTYPED, LVar<allocator_type>(), ty);

      // x' = x + bias
      F::make_binary(
          f(var), EQ,
          F::make_binary(f(from_layer.neurons[i]), ADD, f(to_layer.biases[i])));
    }
    return;
  }

  void make_matmul_node(const Layer &from_layer, const Layer &to_layer) {
    for (size_t j = 0; j < to_layer.size; ++j) {
      // create variable
      auto ty = So(So::Real);
      auto var = F::make_exists(UNTYPED, LVar<allocator_type>(), ty);

      FSeq affine;
      for (size_t i = 0; i < from_layer.size; ++i) {
        affine.push_back(F::make_binary(f(), MUL, f()));
      }
      F linearCons =
          F::make_binary(F::make_nary(ADD, std::move(affine)), EQ, f());
    }

    return;
  }

  void make_gemm_node(const Layer &from_layer, const Layer &to_layer) {
    for (size_t i = 0; i < to_layer.size; ++i) {
      // create variable
      auto ty = So(So::Real);
      auto var = F::make_exists(UNTYPED, LVar<allocator_type>(), ty);

      FSeq affine;
      for (size_t j = 0; j < from_layer.size; ++j) {
        affine.push_back(F::make_binary(f(), MUL, f()));
      }
      affine.push_back(); // bias
      F linearCons =
          F::make_binary(F::make_nary(ADD, std::move(affine)), EQ, f());
    }

    return;
  }

  void make_conv_node(const Layer &from_layer, const Layer &to_layer) {
    make_gemm_node(from_layer, to_layer);
    return;
  }

  void make_constant_node(const Layer &from_layer, const Layer &to_layer) {
    // We do actually create constant node,
    // it would be contained in sub/div node.
    return;
  }

  void make_relu_node(const Layer &from_layer, const Layer &to_layer) {
    for (size_t i = 0; i < to_layer.size; ++i) {
      // create variable
      auto ty = So(So::Real);
      auto var = F::make_exists(UNTYPED, LVar<allocator_type>(), ty);

      // x' = max(0, x)
      F::make_binary(f(var), EQ, f());
    }
    return;
  }

  tensor1d extract1DTensorData(const onnx::TensorProto &tensor) {
    tensor1d data;

    if (tensor.float_data_size() > 0) {
      for (const auto &fd : tensor.float_data()) {
        data.push_back(fd);
      }
    } else {
      std::string raw = tensor.raw_data();
      size_t num_elements = raw.size() / sizeof(float);
      data.resize(num_elements);
      std::memcpy(data.data(), raw.data(), raw.size());
    }

    return data;
  }

  tensor2d extract2DTensorData(const onnx::TensorProto &tensor) {
    tensor1d flat_data = extract1DTensorData(tensor);

    // convert 1D tensor to 2D tensor
    size_t num_rows = tensor.dims(0);
    size_t num_cols = tensor.dims(1);

    // allocate 2D tensor
    tensor2d data(num_cols, tensor1d(num_cols, 0.0f));
    for (size_t r = 0; r < num_rows; ++r) {
      for (size_t c = 0; c < num_cols; ++c) {
        data[r][c] = flat_data[r * num_cols + c];
      }
    }

    return data;
  }

  tensor4d extract4DTensorData(const onnx::TensorProto &tensor) {
    // tensor4d data;
    tensor1d flat_data = extract1DTensorData(tensor);

    // Read dims
    if (tensor.dims_size() != 4)
      throw std::runtime_error("Tensor is not 4D");

    size_t C_out = tensor.dims(0);
    size_t C_in = tensor.dims(1);
    size_t kH = tensor.dims(2);
    size_t kW = tensor.dims(3);

    // Allocate 4D tensor
    tensor4d data(C_out, tensor3d(C_in, tensor2d(kH, tensor1d(kW, 0.0f))));

    // Fill data
    size_t idx = 0;
    for (size_t oc = 0; oc < C_out; ++oc) {
      for (size_t ic = 0; ic < C_in; ++ic) {
        for (size_t h = 0; h < kH; ++h) {
          for (size_t w = 0; w < kW; ++w) {
            data[oc][ic][h][w] = flat_data[idx++];
          }
        }
      }
    }

    return data;
  }

  tensor2d convert4Dto2DTensor(const tensor4d &weights, size_t input_H,
                               size_t input_W, size_t strides, size_t pads) {
    const size_t output_channels = weights.size();
    const size_t input_channels = weights[0].size();
    const size_t kernel_H = weights[0][0].size();
    const size_t kernel_W = weights[0][0][0].size();

    const size_t output_H = (input_H + 2 * pads - kernel_H) / strides + 1;
    const size_t output_W = (input_W + 2 * pads - kernel_W) / strides + 1;

    const size_t input_size = input_channels * input_H * input_W;
    const size_t output_size = output_channels * output_H * output_W;

    tensor2d result_weights(output_size, tensor1d(input_size, 0.0f));
    size_t row = 0;
    for (size_t oc = 0; oc < output_channels; ++oc) {
      for (size_t oh = 0; oh < output_H; ++oh) {
        for (size_t ow = 0; ow < output_W; ++ow) {

          // bounds check for row (defensive)
          if (row >= output_size) {
            std::cerr << "ERROR: row >= output_size: row=" << row
                      << " output_size=" << output_size << std::endl;
            assert(false);
          }

          for (size_t ic = 0; ic < input_channels; ++ic) {
            for (size_t kh = 0; kh < kernel_H; ++kh) {
              for (size_t kw = 0; kw < kernel_W; ++kw) {
                // use signed integers for intermediate coordinates
                int ih = static_cast<int>(oh * strides) + static_cast<int>(kh) -
                         static_cast<int>(pads);
                int iw = static_cast<int>(ow * strides) + static_cast<int>(kw) -
                         static_cast<int>(pads);

                // only write when ih/iw inside input bounds
                if (ih >= 0 && ih < static_cast<int>(input_H) && iw >= 0 &&
                    iw < static_cast<int>(input_W)) {

                  size_t col = ic * (input_H * input_W) +
                               static_cast<size_t>(ih) * input_W +
                               static_cast<size_t>(iw);

                  // defensive check for col bounds
                  if (col >= input_size) {
                    std::cerr << "ERROR: computed col out of range: col=" << col
                              << " input_size=" << input_size
                              << " (oc,ic,oh,ow,kh,kw)=(" << oc << "," << ic
                              << "," << oh << "," << ow << "," << kh << ","
                              << kw << ")\n";
                    assert(false);
                  }

                  // write weight value
                  result_weights[row][col] = weights[oc][ic][kh][kw];
                }
              }
            }
          }
          ++row;
        }
      }
    }

    return result_weights;
  }
};
} // namespace impl
} // namespace lala

#endif
