// Copyright 2025 Yi-Nung Tsao

#ifndef LALA_PARSING_ONNX_PARSER_HPP
#define LALA_PARSING_ONNX_PARSER_HPP

#include <cassert>
#include <cfenv>
#include <cinttypes>
#include <cstdlib>
#include <fstream>
#include <google/protobuf/message_lite.h>
#include <iostream>
#include <istream>
#include <set>
#include <streambuf>
#include <string>
#include <unordered_map>

#include "battery/shared_ptr.hpp"
#include "lala/logic/ast.hpp"
#include "onnx-1.15.0/onnx.proto3.pb.h"
#include "solver_output.hpp"

namespace lala {
namespace impl {

template <class Allocator> class OnnxParser {
  using allocator_type = Allocator;
  using F = TFormula<allocator_type>;
  using So = Sort<allocator_type>;
  using FSeq = typename F::Sequence;

  bool error;  // If an error was found during parsing.
  bool silent; // If we do not want to output error messages.

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
    size_t input_dimensions =
        batch_size * input_channels * input_height * input_width;

    // Build a map from tensor name -> node that produces it
    // This is for linking sub/div node and its input nodes.
    std::unordered_map<std::string, const onnx::NodeProto *> producer_map;
    for (const auto &node : graph.node()) {
      if (node.output_size() > 0) {
        producer_map[node.output(0)] = &node;
      }
    }

    // TODO: Create an additional flatten layer in the biginning.

    // TODO: Iterate over nodes in the network graph
    for (const auto &node : graph.node()) {
      if (node.op_type() == "Sub") {
        continue;
      } else if (node.op_type() == "Div") {
        continue;
      } else if (node.op_type() == "Constant") {
        continue;
      } else if (node.op_type() == "Flatten") {
        continue;
      } else if (node.op_type() == "MatMul") {
        continue;
      } else if (node.op_type() == "Add") {
        continue;
      } else if (node.op_type() == "Gemm") {
        continue;
      } else if (node.op_type() == "Conv") {
        continue;
      } else if (node.op_type() == "Relu") {
        continue;
      }

      // The outputs of the node
      for (const auto &output_name : node.output()) {
        std::cout << "  Output: " << output_name << std::endl;
      }
    }

    google::protobuf::ShutdownProtobufLibrary();

    return;
  }
};
} // namespace impl
} // namespace lala

#endif
