// Copyright 2025 Yi-Nung Tsao

#ifndef LALA_PARSING_CSV_PARSER_HPP
#define LALA_PARSING_CSV_PARSER_HPP

#include <any>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <system_error>

#include "battery/shared_ptr.hpp"
#include "lala/logic/ast.hpp"

#include "lala/solver_output.hpp"
#include "lala/vnnlib_parser.hpp"
#include "lala/onnx_parser.hpp"

namespace lala {
namespace impl {

template<class Allocator>
class NNVParser{
    using allocator_type = Allocator;
    using F = TFormula<allocator_type>;
    using FSeq = typename F::Sequence;
    
    SolverOutput<Allocator>& output;

public:
    NNVParser(SolverOutput<Allocator>& output): output(output) {}

    battery::shared_ptr<F, allocator_type> parse(const std::string& vnnlib_filename, const std::string& onnx_filename){
        FSeq seq;
        std::cout << "Start vnnlib parsring ...\n";
        F vnnlib_formulas = parse_vnnlib(vnnlib_filename, output);
        std::cout << "vnnlib parsing is finished. \n";
        std::cout << "Start onnx parsing ...\n";
        F onnx_formulas = parse_onnx(onnx_filename, output);
        std::cout << "onnx parsing is finished. \n";
        seq.push_back(vnnlib_formulas);
        seq.push_back(onnx_formulas);

        return battery::make_shared<TFormula<Allocator>,Allocator>(std::move(F::make_nary(AND, std::move(seq))));        
    };
};
}

template <class Allocator>
battery::shared_ptr<TFormula<Allocator>, Allocator> parse_nnv(const std::string& vnnlib_filename, const std::string& onnx_filename, SolverOutput<Allocator>& output) {
    impl::NNVParser<Allocator> parser(output);
    return parser.parse(vnnlib_filename, onnx_filename);
}

template <class Allocator>
battery::shared_ptr<TFormula<Allocator>, Allocator> parse_nnv(const std::string& vnnlib_filename, const std::string& onnx_filename, const Allocator& allocator = Allocator()) {
    SolverOutput<Allocator> output(allocator);
    return parse_nnv(vnnlib_filename, onnx_filename, output);
}

}

#endif 