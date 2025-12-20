// Copyright 2025 Yi-Nung Tsao

#ifndef LALA_PARSING_ONNX_PARSER_HPP
#define LALA_PARSING_ONNX_PARSER_HPP

#include "battery/shared_ptr.hpp"
#include "battery/vector.hpp"
#include "lala/logic/ast.hpp"
#include "onnx-1.15.0/onnx.proto3.pb.h"
#include "solver_output.hpp"

#include <google/protobuf/message_lite.h>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <istream>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>

namespace lala {

using tensor1d = battery::vector<float>;
using tensor2d = battery::vector<tensor1d>;
using tensor3d = battery::vector<tensor2d>;
using tensor4d = battery::vector<tensor3d>;

enum class LayerType {
  Input,
  Sub,
  Div,
  Constant,
  MatMul,
  Add,
  Gemm,
  Conv,
  Flatten,
  Relu,
  MaxPool,           
  BatchNormalization, 
  Dropout,            
  Unknown
};

struct Layer {
  size_t size;
  size_t id;
  LayerType type;
  battery::vector<std::string> neurons;
  battery::vector<size_t> source_layers;

  // BatchNormalization
  float epsilon; 
  tensor1d scale; 
  tensor1d B;
  tensor1d input_mean;
  tensor1d input_var;

  // for leaky relu, in GNN, we will need this parameter.
  float alpha; 

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
}  // namespace lala

namespace lala {

namespace impl {

template <class Allocator>
class OnnxParser {
  using allocator_type = Allocator;
  using F = TFormula<allocator_type>;
  using So = Sort<allocator_type>;
  using FSeq = typename F::Sequence;

  battery::vector<Layer> layers;
  bool error;   // If an error was found during parsing.
  bool silent;  // If we do not want to output error messages.

 public:
  OnnxParser() : error(false), silent(false) {}

  F parse(const std::string& onnx_model_directory) {
    onnx::ModelProto model;
    std::ifstream input(onnx_model_directory, std::ios::in | std::ios::binary);

    if (!input) {
      std::cerr << "Failed to open: " + onnx_model_directory << std::endl;
      error = true;
      return F::make_false();
    }

    if (!model.ParseFromIstream(&input)) {
      std::cerr << "Failed to parse onnx file." << std::endl;
      error = true;
      return F::make_false();
    }

    const onnx::GraphProto& graph = model.graph();
#ifndef NDEBUG
    std::cout << "onnx model name: " << graph.name() << std::endl;
    std::cout << "number of nodes: " << graph.node_size() << std::endl;
    std::cout << "number of initializers (weights/biases): " << graph.initializer_size() << std::endl;
#endif

    // Create a map from tensor name to TensorProto for fast lookup
    std::unordered_map<std::string, onnx::TensorProto> tensor_map;
    for (const auto& initializer : graph.initializer()) {
      tensor_map[initializer.name()] = initializer;
    }

    // Shape of input.
    const onnx::ValueInfoProto& graph_input = graph.input(0);
    const auto& input_shape = graph_input.type().tensor_type().shape();  // <batch_size, num_input_channels, input_H, input_W>;
    size_t batch_size = input_shape.dim(0).dim_value();
    size_t input_channels = input_shape.dim().size() > 1 ? input_shape.dim(1).dim_value() : 1;
    size_t input_height = input_shape.dim().size() > 2 ? input_shape.dim(2).dim_value() : 1;
    size_t input_width = input_shape.dim().size() > 3 ? input_shape.dim(3).dim_value() : 1;
    size_t input_dimensions = batch_size * input_channels * input_height * input_width;  // number of input neurons.

    // Build a map from tensor name -> node that produces it
    // This is for linking sub/div node and its input nodes.
    std::unordered_map<std::string, const onnx::NodeProto*> producer_map;
    for (const auto& node : graph.node()) {
      if (node.output_size() > 0) {
        producer_map[node.output()[0]] = &node;
      }
    }

    FSeq seq;
    // Create an input layer.
    layers.reserve(graph.node_size());
    Layer input_layer;
    input_layer.id = layers.size();
    input_layer.type = LayerType::Input;
    input_layer.size = input_dimensions;
    input_layer.input_height = input_height;
    input_layer.input_width = input_width;
    seq.push_back(std::move(make_input_node(input_layer)));
    layers.emplace_back(input_layer);

    // Iterate over nodes in the network graph
    std::unordered_map<std::string, size_t> layer_index_map; // key: input node name, value: its layer index
    layer_index_map[graph.input()[0].name()] = input_layer.id;
    for (const auto& node : graph.node()) {
      Layer layer;
      layer.id = layers.size();
      layer.type = setLayerType(node);
#ifndef NDEBUG
      std::cout << "Node: " << node.output()[0] << "| OpType: " << node.op_type() << std::endl;
#endif
      if (layer.type == LayerType::Unknown) {
        std::cerr << "Unknown layer type." << std::endl;
        error = true;
        return F::make_false();
      } 
      else if (layer.type == LayerType::Constant) { continue; }

      // Note: in some networks, there is no node name, but the output name is always well defined. 
      //       So we use output name to match the layer index except input layer.
      layer_index_map[node.output()[0]] = layer.id; 
      for (const auto& input_name : node.input()) {
        // If the input is a weight tensor
        if (tensor_map.find(input_name) != tensor_map.end()) {
          // In this case, node.op_type() could be Gemm, MatMul, Add, Conv
          const auto& tensor = tensor_map[input_name];
          if (tensor.dims().size() == 1) {
            // bias 1d tensor
            layer.biases = extract1DTensorData(tensor);
            updateBiasTensor(layer);
          } 
          else if (tensor.dims().size() == 2) {
            // weight 2d tensor
            layer.weights = extract2DTensorData(tensor);
            layer.size = layer.weights[0].size();  // <output_dimensions, input_dimensions>
          } 
          else if (tensor.dims().size() == 4 && layer.type == LayerType::Conv) {
            extractIntAttributes(node, layer);
            layer.conv_weights = extract4DTensorData(tensor);
            // layer.weights = convert4Dto2DTensor(layer.conv_weights, input_height, input_width, layer.strides, layer.pads);
            updateConvMaxPoolLayerInfo(layer, input_height, input_width);
          }
          else { 
            std::cerr << "Unknown behavior!" << std::endl; 
            error = true;
            return F::make_false();
          }
        } 
        else if (layer.type == LayerType::Sub) {
          // Finding the constant in Sub node
          if (producer_map.find(input_name) != producer_map.end()) {
            if (extractConstant(producer_map[input_name], layer.sub_values))
              continue;
          }
          layer.size = layers[0].size;
          layer.input_height = layers[0].input_height;
          layer.input_width = layers[0].input_width;
          layer.source_layers.push_back(layer_index_map[input_name]);
        } 
        else if (layer.type == LayerType::Div) {
          // Finding the constant in Div node
          if (producer_map.find(input_name) != producer_map.end()) {
            if (extractConstant(producer_map[input_name], layer.div_values))
              continue;
          }
          layer.size = layers[0].size; 
          layer.input_height = layers[0].input_height;
          layer.input_width = layers[0].input_width;
          layer.source_layers.push_back(layer_index_map[input_name]);
        }
        else if (layer.type == LayerType::Flatten) { 
          layer.size = layers[layers.size() - 1].size; 
          layer.source_layers.push_back(layer_index_map[input_name]);
        } 
        else if (layer.type == LayerType::Relu) { 
          layer.size = layers[layers.size() - 1].size; 
          layer.source_layers.push_back(layer_index_map[input_name]);
        }
        else if (layer.type == LayerType::MaxPool) { 
          updateConvMaxPoolLayerInfo(layer, input_height, input_width); 
          layer.source_layers.push_back(layer_index_map[input_name]);
        } 
        else if (layer.type == LayerType::Gemm) { layer.source_layers.push_back(layer_index_map[input_name]); }
        else if (layer.type == LayerType::MatMul) { layer.source_layers.push_back(layer_index_map[input_name]); }
        else if (layer.type == LayerType::Conv) { layer.source_layers.push_back(layer_index_map[input_name]); }
        else if (layer.type == LayerType::Add) { 
          layer.size = layers[layers.size() - 1].size;
          layer.source_layers.push_back(layer_index_map[input_name]); 
        }  
        else if (layer.type == LayerType::Dropout) { 
          layer.size = layers[layers.size() - 1].size; 
          layer.source_layers.push_back(layer_index_map[input_name]);
        }
        else if (layer.type == LayerType::BatchNormalization){ 
          layer.size = layers[layers.size() - 1].size; 
          layer.source_layers.push_back(layer_index_map[input_name]);
        } 
        else { 
          std::cerr << "Unknown layer type." << std::endl; 
          error = true;
          return F::make_false(); 
        }
      }

#ifndef NDEBUG
      // The outputs of the node
      for (const auto& output_name : node.output()) {
        std::cout << "  Output: " << output_name << std::endl;
      }
#endif

      // create variables at current layer then build constraint for this node/layer
      make_neurons(layer, graph.output()[0].name() == node.output()[0].c_str());
      seq.push_back(std::move(make_node(layer)));
      // store into list of layers
      layers.emplace_back(layer);
    }

    google::protobuf::ShutdownProtobufLibrary();

    return F::make_nary(AND, std::move(seq));
  }

 private:
  static F f(const std::any& any) { return std::any_cast<F>(any); }

  LayerType setLayerType(const onnx::NodeProto& node) {
    if (node.op_type() == "Sub") { return LayerType::Sub; } 
    else if (node.op_type() == "Div") { return LayerType::Div; } 
    else if (node.op_type() == "Constant") { return LayerType::Constant; } 
    else if (node.op_type() == "Flatten") { return LayerType::Flatten; } 
    else if (node.op_type() == "MatMul") { return LayerType::MatMul; } 
    else if (node.op_type() == "Add") { return LayerType::Add; } 
    else if (node.op_type() == "Gemm") { return LayerType::Gemm; } 
    else if (node.op_type() == "Conv") { return LayerType::Conv; } 
    else if (node.op_type() == "Relu") { return LayerType::Relu; } 
    else if (node.op_type() == "MaxPool") { return LayerType::MaxPool; }
    else if (node.op_type() == "Dropout") { return LayerType::Dropout; }
    else if (node.op_type() == "BatchNormalization") { return LayerType::BatchNormalization; }
    else { return LayerType::Unknown; }
  }

  F make_node(const Layer& layer) {
    if (layer.type == LayerType::Sub) { return make_sub_node(layer); } 
    else if (layer.type == LayerType::Div) { return make_div_node(layer); } 
    else if (layer.type == LayerType::Constant) { return make_constant_node(layer); } 
    else if (layer.type == LayerType::Flatten) { return make_flatten_node(layer); } 
    else if (layer.type == LayerType::MatMul) { return make_matmul_node(layer); } 
    else if (layer.type == LayerType::Add) { return make_add_node(layer); } 
    else if (layer.type == LayerType::Gemm) { return make_gemm_node(layer); } 
    else if (layer.type == LayerType::Conv) { return make_conv_node(layer); } 
    else if (layer.type == LayerType::Relu) { return make_relu_node(layer); } 
    else if (layer.type == LayerType::MaxPool) { return make_maxpool_node(layer); }
    else if (layer.type == LayerType::Dropout) { return make_dropout_node(layer); }
    else if (layer.type == LayerType::BatchNormalization) { return make_batch_normalization_node(layer); }
    else { return F::make_false(); }
  }

  void make_neurons(Layer& layer, bool isOutputLayer) {
    if (isOutputLayer){
      // NOTE: rename the variables at the output layer as starting by Y
      for (size_t i = 0; i < layer.size; ++i) {
        layer.neurons.push_back("Y_" + std::to_string(i));
      }
    }
    else {
      for (size_t i = 0; i < layer.size; ++i) {
        layer.neurons.push_back("X_" + std::to_string(layer.id) + "_" + std::to_string(i + 1));
      }
    }
    return;
  }

  F make_input_node(Layer &input_layer) {
    FSeq seq; 
    for (size_t i = 0; i < input_layer.size; ++i) { 
      input_layer.neurons.push_back("X_" + std::to_string(i));
      auto var = F::make_exists(UNTYPED, LVar<allocator_type>(input_layer.neurons[i]), So(So::Real));
      seq.push_back(std::move(var));
    }

    return F::make_nary(AND, std::move(seq));
  }

  F make_sub_node(const Layer& layer) {
    assert(layer.source_layers.size() == 1);
    FSeq seq; 
    if (layer.sub_values.size() == 0) {
      std::cout << "layer.size = " << layer.size << std::endl;
      std::cout << "source layer size = " << layers[layer.source_layers[0]].size << std::endl;
      std::cout << "source layer id = " << layer.source_layers[0] << std::endl;
      for (size_t i = 0; i < layer.size; ++i) {
        // create variable
        auto var = F::make_exists(UNTYPED, LVar<allocator_type>(layer.neurons[i]), So(So::Real));
        seq.push_back(std::move(var));

        seq.push_back(F::make_binary(
          F::make_lvar(UNTYPED, LVar<allocator_type>(layer.neurons[i])),
          EQ,
          F::make_lvar(UNTYPED, LVar<allocator_type>(layers[layer.source_layers[0]].neurons[i]))));
      }
    }
    else {
      std::cout << "layer.size = " << layer.size << std::endl;
      std::cout << "source layer id = " << layer.source_layers[0] << std::endl;
      std::cout << "source layer size = " << layers[layer.source_layers[0]].size << std::endl;
      for (size_t i = 0; i < layer.size; ++i) {
        int dim = i / (layer.input_height * layer.input_width);

        // create variable 
        auto var = F::make_exists(UNTYPED, LVar<allocator_type>(layer.neurons[i]), So(So::Real));
        seq.push_back(std::move(var));

        // x' = x - sub_value;
        seq.push_back(F::make_binary(
            F::make_lvar(UNTYPED, LVar<allocator_type>(layer.neurons[i])),
            EQ,
            std::move(F::make_binary(F::make_lvar(UNTYPED, LVar<allocator_type>(layers[layer.source_layers[0]].neurons[i])),
                           SUB,
                           F::make_real(layer.sub_values[dim], layer.sub_values[dim])))));
      }
    }
    return F::make_nary(AND, std::move(seq));
  }

  F make_div_node(const Layer& layer) {
    assert(layer.source_layers.size() == 1);
    FSeq seq;
    if (layer.div_values.size() == 0) {
      for (size_t i = 0; i < layer.size; ++i) {
        // create variable
        auto var = F::make_exists(UNTYPED, LVar<allocator_type>(layer.neurons[i]), So(So::Real));
        seq.push_back(std::move(var));

        seq.push_back(F::make_binary(
            F::make_lvar(UNTYPED, LVar<allocator_type>(layer.neurons[i])),
            EQ,
            F::make_lvar(UNTYPED, LVar<allocator_type>(layers[layer.source_layers[0]].neurons[i]))));
      }
    } 
    else {
      for (size_t i = 0; i < layer.size; ++i) {
        int dim = i / (layer.input_height * layer.input_width);

        // create variable
        auto var = F::make_exists(UNTYPED, LVar<allocator_type>(layer.neurons[i]), So(So::Real));
        seq.push_back(std::move(var));

        // x' = x / div_value;
        seq.push_back(F::make_binary(
            F::make_lvar(UNTYPED, LVar<allocator_type>(layer.neurons[i])),
            EQ,
            std::move(F::make_binary(F::make_lvar(UNTYPED, LVar<allocator_type>(layers[layer.source_layers[0]].neurons[i])),
                           DIV,
                           F::make_real(layer.div_values[dim], layer.div_values[dim])))));
      }
    }
    return F::make_nary(AND, std::move(seq));
  }

  F make_flatten_node(const Layer& layer) {
    assert(layer.source_layers.size() == 1);
    // we have flatten input as a vector. we do not need to do additional transformation here.
    FSeq seq;
    for (size_t i = 0; i < layer.size; ++i) {
      // create variable
      auto var = F::make_exists(UNTYPED,LVar<allocator_type>(layer.neurons[i]), So(So::Real));
      seq.push_back(std::move(var));

      seq.push_back(F::make_binary(
          F::make_lvar(UNTYPED,LVar<allocator_type>(layer.neurons[i])), 
          EQ,
          F::make_lvar(UNTYPED,LVar<allocator_type>(layers[layer.source_layers[0]].neurons[i]))));
    }

    return F::make_nary(AND, std::move(seq));
  }

  F make_add_node(const Layer& layer) {
    assert(layer.source_layers.size() >= 1);
    FSeq seq;
    for (size_t i = 0; i < layer.size; ++i) {
      // create variable 
      auto var = F::make_exists(UNTYPED, LVar<allocator_type>(layer.neurons[i]), So(So::Real));
      seq.push_back(std::move(var));

      FSeq rhs; 
      for (size_t sidx = 0; sidx < layer.source_layers.size(); ++sidx) {
        // not sure which one is correct yet.
        // for (size_t j = 0; j < layers[layer.source_layers[sidx]].size; ++j){
        //   rhs.push_back(F::make_lvar(UNTYPED, LVar<allocator_type>(layers[layer.source_layers[sidx]].neurons[j])));
        // }
        rhs.push_back(F::make_lvar(UNTYPED, LVar<allocator_type>(layers[layer.source_layers[sidx]].neurons[i])));
      }
      rhs.push_back(F::make_real(layer.biases[i], layer.biases[i]));

      seq.push_back(F::make_binary(
        F::make_lvar(UNTYPED, LVar<allocator_type>(layer.neurons[i])),
        EQ, 
        F::make_nary(ADD, std::move(rhs))));
    }

    return F::make_nary(AND, std::move(seq));
  }

  F make_matmul_node(const Layer& layer) {
    assert(layer.source_layers.size() == 1);
    FSeq seq;
    for (size_t j = 0; j < layer.size; ++j) {
      // create variable
      auto var = F::make_exists(UNTYPED,LVar<allocator_type>(layer.neurons[j]), So(So::Real));
      seq.push_back(std::move(var));

      FSeq affine;
      for (size_t i = 0; i < layers[layer.source_layers[0]].size; ++i) {
        affine.push_back(F::make_binary(
            F::make_real(layer.weights[i][j], layer.weights[i][j]),
            MUL,
            F::make_lvar(UNTYPED,LVar<allocator_type>(layers[layer.source_layers[0]].neurons[i]))));
      }
      seq.push_back(F::make_binary(
          F::make_lvar(UNTYPED,LVar<allocator_type>(layer.neurons[j])), 
          EQ,
          F::make_nary(ADD, std::move(affine))));
    }

    return F::make_nary(AND, std::move(seq));
  }

  F make_gemm_node(const Layer& layer) {
    assert (layer.source_layers.size() == 1);
    FSeq seq;
    for (size_t i = 0; i < layer.size; ++i) {
      // create variable
      auto var = F::make_exists(UNTYPED, LVar<allocator_type>(layer.neurons[i]), So(So::Real));
      seq.push_back(std::move(var));

      FSeq affine;
      for (size_t j = 0; j < layers[layer.source_layers[0]].size; ++j) {
        affine.push_back(F::make_binary(
            F::make_real(layer.weights[i][j], layer.weights[i][j]),
            MUL,
            F::make_lvar(UNTYPED,LVar<allocator_type>(layers[layer.source_layers[0]].neurons[j]))));
      }
      affine.push_back(F::make_real(layer.biases[i], layer.biases[i]));  // bias
      seq.push_back(F::make_binary(F::make_lvar(UNTYPED, LVar<allocator_type>(layer.neurons[i])), 
                                    EQ,
                                    F::make_nary(ADD, std::move(affine))));
    }

    return F::make_nary(AND, std::move(seq));
  }
  
  F make_conv_node(const Layer& layer) {
    // This function is super slow and takes a lot of memory space, not sure the reason yet.
    assert(layer.source_layers.size() == 1);
    const size_t input_size = layer.input_channels * layer.conv_input_height * layer.conv_input_width;
    const size_t output_size = layer.output_channels * layer.conv_output_height * layer.conv_output_width;

    FSeq seq;
    // seq.reserve(output_size * 2);
    size_t row = 0;
    for (size_t oc = 0; oc < layer.output_channels; ++oc) {
      for (size_t oh = 0; oh < layer.conv_output_height; ++oh) {
        for (size_t ow = 0; ow < layer.conv_output_width; ++ow) {
#ifndef NDEBUG
          // bound check for now (defensive)
          if (row >= output_size) {
            std::cerr << "ERROR: row >= output_size: row=" << row << "output_size=" << output_size << std::endl;
            assert(false);
          }
#endif
          // create variable
          auto var = F::make_exists(UNTYPED, LVar<allocator_type>(layer.neurons[row]), So(So::Real));
          seq.push_back(std::move(var));

          FSeq affine;
          affine.reserve(layer.input_channels * layer.kernel_height * layer.kernel_width + 1);
          for (size_t ic = 0; ic < layer.input_channels; ++ic) {
            for (size_t kh = 0; kh < layer.kernel_height; ++kh) {
              for (size_t kw = 0; kw < layer.kernel_width; ++kw) {
                // use signed integers for intermediate coordinates. 
                int ih = oh * layer.strides + kh - layer.pads;
                int iw = ow * layer.strides + kw - layer.pads; 

                if (ih >= 0 && ih < layer.conv_input_height && iw >= 0 && iw < layer.conv_input_width) {
                  size_t col = ic * (layer.conv_input_height * layer.conv_input_width) + ih * layer.conv_input_width + iw;
#ifndef NDEBUG
                  // defensive check for col bounds
                  if (col >= input_size) {
                    std::cerr << "ERROR: computed col out of range: col=" << col << " input_size=" << input_size 
                              << " (oc,ic,oh,ow,kh,kw)=(" << oc << "," << ic << "," << oh << "," << ow << "," << kh << "," << kw << ")\n";
                    assert(false);                    
                  }
#endif 
                  affine.emplace_back(F::make_binary(
                    F::make_real(layer.conv_weights[oc][ic][kh][kw], layer.conv_weights[oc][ic][kh][kw]),
                    MUL, 
                    F::make_lvar(UNTYPED, LVar<allocator_type>(layers[layer.source_layers[0]].neurons[col]))));
                }
              }
            }
          }
          affine.emplace_back(F::make_real(layer.biases[row], layer.biases[row]));
          seq.push_back(F::make_binary(F::make_lvar(UNTYPED, LVar<allocator_type>(layer.neurons[row])), EQ, F::make_nary(ADD, std::move(affine))));
          ++row;
        }
      }
    }

    return F::make_nary(AND, std::move(seq));
  }

  // We do actually create constant node,
  // it would be contained in sub/div node.
  F make_constant_node(const Layer& layer) { return F::make_true(); }

  F make_relu_node(const Layer& layer) {
    assert(layer.source_layers.size() == 1);
    FSeq seq;
    for (size_t i = 0; i < layer.size; ++i) {
      // create variable
      auto var = F::make_exists(UNTYPED,LVar<allocator_type>(layer.neurons[i]), So(So::Real));
      seq.push_back(std::move(var));

      // x' = max(0, x)
      seq.push_back(F::make_binary(
          F::make_lvar(UNTYPED,LVar<allocator_type>(layer.neurons[i])), 
          EQ,
          F::make_binary(
              F::make_lvar(UNTYPED,LVar<allocator_type>(layers[layer.source_layers[0]].neurons[i])),
              MAX,
              F::make_z(0))));
    }
    return F::make_nary(AND, std::move(seq));
  }

  F make_maxpool_node(const Layer& layer) {
    // This function is super slow and takes a lot of memory space, not sure the reason yet.
    assert(layer.source_layers.size() == 1);
    const size_t input_size = layer.input_channels * layer.conv_input_height * layer.conv_input_width;
    const size_t output_size = layer.output_channels * layer.conv_output_height * layer.conv_output_width;

    FSeq seq;
    seq.reserve(output_size * 2);
    size_t row = 0;
    for (size_t oc = 0; oc < layer.output_channels; ++oc) {
      for (size_t oh = 0; oh < layer.conv_output_height; ++oh) {
        for (size_t ow = 0; ow < layer.conv_output_width; ++ow) {
#ifndef NDEBUG
          // bound check for now (defensive)
          if (row >= output_size) {
            std::cerr << "ERROR: row >= output_size: row=" << row << "output_size=" << output_size << std::endl;
            assert(false);
          }
#endif
          // create variable
          auto var = F::make_exists(UNTYPED, LVar<allocator_type>(layer.neurons[row]), So(So::Real));
          seq.emplace_back(std::move(var));

          FSeq max_seq;
          for (size_t ic = 0; ic < layer.input_channels; ++ic) {
            for (size_t kh = 0; kh < layer.kernel_height; ++kh) {
              for (size_t kw = 0; kw < layer.kernel_width; ++kw) {
                // use signed integers for intermediate coordinates. 
                int ih = oh * layer.strides + kh - layer.pads;
                int iw = ow * layer.strides + kw - layer.pads; 

                if (ih >= 0 && ih < layer.conv_input_height && iw >= 0 && iw < layer.conv_input_width) {
                  size_t col = ic * (layer.conv_input_height * layer.conv_input_width) + ih * layer.conv_input_width + iw;
#ifndef NDEBUG
                  // defensive check for col bounds
                  if (col >= input_size) {
                    std::cerr << "ERROR: computed col out of range: col=" << col << " input_size=" << input_size 
                              << " (oc,ic,oh,ow,kh,kw)=(" << oc << "," << ic << "," << oh << "," << ow << "," << kh << "," << kw << ")\n";
                    assert(false);                    
                  }
#endif 
                  max_seq.push_back(F::make_lvar(UNTYPED, LVar<allocator_type>(layers[layer.source_layers[0]].neurons[col])));
                }
              }
            }
          }
          seq.emplace_back(F::make_binary(F::make_lvar(UNTYPED, LVar<allocator_type>(layer.neurons[row])), EQ, F::make_nary(MAX, std::move(max_seq))));
          ++row;
        }
      }
    }

    return F::make_nary(AND, std::move(seq));
  }

  F make_dropout_node(const Layer& layer) {
    assert (layer.source_layers.size() == 1);
    FSeq seq;
    for (size_t i = 0; i < layer.size; ++i) {
      auto var = F::make_exists(UNTYPED, LVar<allocator_type>(layer.neurons[i]), So(So::Real));
      seq.push_back(std::move(var));
      
      // x = y;
      seq.push_back(F::make_binary(
        F::make_lvar(UNTYPED, LVar<allocator_type>(layer.neurons[i])),
        EQ, 
        F::make_lvar(UNTYPED, LVar<allocator_type>(layers[layer.source_layers[0]].neurons[i]))));
    }

    return F::make_nary(AND, std::move(seq));
  }

  F make_batch_normalization_node(const Layer& layer) {
    assert(layer.source_layers.size() == 1);
    FSeq seq;
    for (size_t i = 0; i < layer.size; ++i) {
      auto var = F::make_exists(UNTYPED, LVar<allocator_type>(layer.neurons[i]), So(So::Real));
      seq.push_back(std::move(var));

      FSeq batch_norm;
      batch_norm.push_back(F::make_binary(F::make_lvar(UNTYPED, LVar<allocator_type>(layers[layer.source_layers[0]].neurons[i])), 
                          MUL, 
                          F::make_real(1 / sqrt(layer.input_var[i] + layer.epsilon) * layer.scale[i], 1 / sqrt(layer.input_var[i] + layer.epsilon) * layer.scale[i])));
      batch_norm.push_back(F::make_real(layer.input_mean[i] / sqrt(layer.input_var[i] + layer.epsilon) * layer.scale[i] - layer.biases[i], layer.input_mean[i] / sqrt(layer.input_var[i] + layer.epsilon) * layer.scale[i] - layer.biases[i]));
      
      seq.push_back(F::make_binary(
        F::make_lvar(UNTYPED, LVar<allocator_type>(layer.neurons[i])),
        EQ, 
        F::make_nary(ADD, std::move(batch_norm))   
      ));
    }
    return F::make_nary(AND, std::move(seq));
  }

  void extractIntAttributes(const onnx::NodeProto& node, Layer& layer) {
    for (const auto& attr : node.attribute()) {
      if (attr.ints_size() > 0) {
        const std::string& attr_name = attr.name();
        if (attr_name == "dilations") { layer.dilations = attr.ints()[0]; }
        else if (attr_name == "group") { layer.group = attr.ints()[0]; }
        else if (attr_name == "kernel_shape") {
          layer.kernel_height = attr.ints()[0];
          layer.kernel_width = attr.ints()[0];
        }
        else if (attr_name == "pads") { layer.pads = attr.ints()[0]; }
        else if (attr_name == "strides") { layer.strides = attr.ints()[0]; }
        else { std::cout << "Unknown int attribute " << std::endl; }
      }
    }

    return;
  }

  void extractFloatAttributes(const onnx::NodeProto* node, Layer& layer) {
    for (const auto& attr : node->attribute()) {
      if (attr.floats_size() > 0) {
        const std::string& attr_name = attr.name();
        if (attr_name == "alpha") { layer.alpha = attr.floats()[0]; }
        else if (attr_name == "epsilon") { layer.epsilon = attr.floats()[0]; }
        else { std::cout << "Unknown float attribute" << std::endl; }
      }
    } 

    return;
  }

  bool extractConstant(const onnx::NodeProto* node, tensor1d& constant_tensor) {
    bool hasAttributes = false;
    for (const auto& attr : node->attribute()) {
      hasAttributes = true;
      if (attr.has_t()) {
        const onnx::TensorProto& tensor = attr.t();
        if (tensor.data_type() == onnx::TensorProto::FLOAT) {
          if (!tensor.raw_data().empty()) {
            const std::string& raw = tensor.raw_data();
            const float* data = reinterpret_cast<const float*>(raw.data());
            size_t numel = raw.size() / sizeof(float);
            for (size_t i = 0; i < numel; ++i) {
              constant_tensor.push_back(data[i]);
            }
          }
        }
      }
      else if (attr.floats_size() > 0) {
        for (const auto& f : attr.floats()) {
          constant_tensor.push_back(f);
        }
      }
      else { std::cout << "Not implemented yet." << std::endl; }
    }

    std::cout << "finished attributes extraction procedure.\n";

    return hasAttributes;
  }

  tensor1d extract1DTensorData(const onnx::TensorProto& tensor) {
    tensor1d data;

    if (tensor.float_data_size() > 0) {
      for (const auto& fd : tensor.float_data()) {
        data.push_back(fd);
      }
    } 
    else {
      std::string raw = tensor.raw_data();
      size_t num_elements = raw.size() / sizeof(float);
      data.resize(num_elements);
      std::memcpy(data.data(), raw.data(), raw.size());
    }

    return data;
  }

  void updateBiasTensor(Layer &layer) {
    if (layer.type == LayerType::Gemm || layer.type == LayerType::Add) {
      layer.size = layer.biases.size();
    }
    else if (layer.type == LayerType::Conv) {
      tensor1d expanded_biases; 
      expanded_biases.reserve(layer.size);
      size_t spatial_size = layer.conv_output_height * layer.conv_output_width;
      for (size_t i = 0; i < layer.biases.size(); ++i) {
        for (size_t j = 0; j < spatial_size; ++j) {
          expanded_biases.emplace_back(layer.biases[i]);
        }
      }
      layer.biases = expanded_biases;
    }

    return;
  }

  void updateConvMaxPoolLayerInfo(Layer &layer, size_t &input_height, size_t &input_width) {
    // <output_dim, input_dim, kernel_height, kernel_weight>
    layer.output_channels = layer.conv_weights.size();
    layer.input_channels = layer.conv_weights[0].size();
    layer.conv_input_height = input_height;
    layer.conv_input_width = input_width;
    layer.kernel_height = layer.conv_weights[0][0].size();
    layer.kernel_width = layer.conv_weights[0][0][0].size();
    layer.conv_output_height = (input_height + 2 * layer.pads - layer.kernel_height) / layer.strides + 1;
    layer.conv_output_width = (input_width + 2 * layer.pads - layer.kernel_width) / layer.strides + 1;
    
    layer.size = layer.output_channels * layer.conv_output_height * layer.conv_output_width;
    input_height = std::floor((layer.conv_input_height + 2 * layer.pads - layer.dilations * (layer.kernel_height - 1) - 1) / layer.strides + 1);
    input_width = std::floor((layer.conv_input_width + 2 * layer.pads - layer.dilations * (layer.kernel_width - 1) - 1) / layer.strides + 1);
  
    return;
  }

  tensor2d extract2DTensorData(const onnx::TensorProto& tensor) {
    tensor1d flat_data = extract1DTensorData(tensor);

    // convert 1D tensor to 2D tensor
    size_t num_rows = tensor.dims(0);
    size_t num_cols = tensor.dims(1);

    // allocate 2D tensor
    tensor2d data(num_rows, tensor1d(num_cols, 0.0f));
    for (size_t r = 0; r < num_rows; ++r) {
      for (size_t c = 0; c < num_cols; ++c) {
        data[r][c] = flat_data[r * num_cols + c];
      }
    }

    return data;
  }

  tensor4d extract4DTensorData(const onnx::TensorProto& tensor) {
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

  tensor2d convert4Dto2DTensor(const tensor4d& weights, size_t input_H, size_t input_W, size_t strides, size_t pads) {
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
                int ih = static_cast<int>(oh * strides) + static_cast<int>(kh) - static_cast<int>(pads);
                int iw = static_cast<int>(ow * strides) + static_cast<int>(kw) - static_cast<int>(pads);

                // only write when ih/iw inside input bounds
                if (ih >= 0 && ih < static_cast<int>(input_H) && iw >= 0 && iw < static_cast<int>(input_W)) {
                  size_t col = ic * (input_H * input_W) + static_cast<size_t>(ih) * input_W + static_cast<size_t>(iw);

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
    std::cout << "after converting 4d tensor to 2d tensor\n";
    return result_weights;
  }
};
}  // namespace impl

template <class Allocator>
TFormula<Allocator> parse_onnx_str(const std::string& input) {
  impl::OnnxParser<Allocator> parser;
  return parser.parse(input);
}

template <class Allocator>
TFormula<Allocator> parse_onnx(const std::string& filename) {
  return parse_onnx_str<Allocator>(filename);
}
}  // namespace lala

#endif
