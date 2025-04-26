// Copyright 2024 Thibault Falque, Pierre Talbot

#ifndef LALA_PARSING_SOLVER_OUTPUT_HPP
#define LALA_PARSING_SOLVER_OUTPUT_HPP

#include <functional>
#include <peglib.h>
#include <lala/logic/ast.hpp>

namespace lala {

enum class OutputType {
  XCSP,
  FLATZINC
};

template<class Allocator>
class SolverOutput {
  using bstring = battery::string<Allocator>;
  template<class T> using bvector = battery::vector<T, Allocator>;
  using array_dim_t = bvector<battery::tuple<size_t,size_t>>;
  using F = TFormula<Allocator>;

  bvector<bstring> output_vars;
  // For each array, we store its output dimension characteristics and the list of the variables in the array.
  // We also encode sets in the same vector (the Boolean is `true` if it is a set variable).
  bvector<battery::tuple<bstring, bool, array_dim_t, bvector<bstring>>> output_arrays;

  OutputType type;
  std::string join_str(const bvector<bstring>& vec, const std::string& separator,  std::function<std::string(const bstring&)> toString) const {
    std::string result;
    for (size_t i = 0; i < vec.size(); ++i) {
      result += toString(vec[i]);
      if (i < vec.size() - 1) {
        result += separator;
      }
    }
    return result;
  }

public:
  template <class Alloc2>
  friend class SolverOutput;

  CUDA SolverOutput(const Allocator& alloc)
    : output_vars(alloc)
    , output_arrays(alloc),type(OutputType::FLATZINC)
  {}

  CUDA SolverOutput(const Allocator& alloc,OutputType outputType)
      : output_vars(alloc)
      , output_arrays(alloc),type(outputType)
  {}

  SolverOutput(SolverOutput&&) = default;
  SolverOutput<Allocator>& operator=(const SolverOutput<Allocator>&) = default;

  template <class Alloc>
  CUDA SolverOutput<Allocator>& operator=(const SolverOutput<Alloc>& other) {
    output_vars = other.output_vars;
    output_arrays = other.output_arrays;
    type = other.type;
    return *this;
  }

  template<class Alloc2>
  CUDA SolverOutput(const SolverOutput<Alloc2>& other, const Allocator& allocator = Allocator{})
    : output_vars(other.output_vars, allocator)
    , output_arrays(other.output_arrays, allocator),type(other.type)
  {}

  void add_array_var(const std::string& name, const bstring& var_name, const peg::SemanticValues& sv) {
    int idx = -1;
    auto array_name = bstring(name.data());
    for(int i = 0; i < output_arrays.size(); ++i) {
      if(battery::get<0>(output_arrays[i]) == array_name) {
        idx = i;
        break;
      }
    }
    if(idx == -1) {
      output_arrays.push_back(battery::make_tuple<bstring, bool, array_dim_t, bvector<bstring>>(bstring(array_name), false, {}, {}));
      idx = static_cast<int>(output_arrays.size()) - 1;
      // Add the dimension of the array.
      for(int i = 0; i < sv.size(); ++i) {
        auto range = std::any_cast<F>(sv[i]);
        for(int j = 0; j < range.s().size(); ++j) {
          const auto& itv = range.s()[j];
          battery::get<2>(output_arrays[idx]).push_back(battery::make_tuple(battery::get<0>(itv).z(), battery::get<1>(itv).z()));
        }
      }
    }
    battery::get<3>(output_arrays[idx]).push_back(var_name);
  }

  void add_var(const bstring& var_name) {
    output_vars.push_back(var_name);
  }

  class SimplifierIdentity {
    template <class Alloc, class B, class Env>
    CUDA void print_variable(const LVar<Alloc>& vname, const Env& benv, const B& b) const {
      const auto& x = *(benv.variable_of(vname));
      x.sort.print_value(b.project(x.avars[0]));
    }
  };

  template <class Env, class A, class S>
  CUDA void print_solution_flatzinc(const Env& env, const A& sol, const S& simplifier = SimplifierIdentity{}) const {
    for(int i = 0; i < output_vars.size(); ++i) {
      printf("%s=", output_vars[i].data());
      simplifier.print_variable(output_vars[i], env, sol);
      printf(";\n");
    }
    for(int i = 0; i < output_arrays.size(); ++i) {
      const auto& dims = battery::get<2>(output_arrays[i]);
      const auto& array_vars = battery::get<3>(output_arrays[i]);
      printf("%s=array%" PRIu64 "d(", battery::get<0>(output_arrays[i]).data(), dims.size());
      for(int j = 0; j < dims.size(); ++j) {
        printf("%" PRIu64 "..%" PRIu64 ",", battery::get<0>(dims[j]), battery::get<1>(dims[j]));
      }
      printf("[");
      for(int j = 0; j < array_vars.size(); ++j) {
        simplifier.print_variable(array_vars[j], env, sol);
        if(j+1 != array_vars.size()) {
          printf(",");
        }
      }
      printf("]);\n");
    }
  }

  template<class Env, class A, class S>
  CUDA void print_solution_xml(const Env& env, const A& sol, const S& simplifier = SimplifierIdentity{}) const {
    auto vars = join_str(output_vars, " ", [](const bstring& s) -> std::string { return s.data(); });
    printf("v <instantiation> <list>%s</list> <values>", vars.c_str());
    for (int i = 0; i < output_vars.size(); ++i) {
      simplifier.print_variable(output_vars[i], env, sol);
      if(i+1 != output_vars.size()) {
        printf(" ");
      }
    }
    printf("</values> </instantiation>\n");
  }

  template <class Env, class A, class S>
  CUDA void print_solution(const Env& env, const A& sol, const S& simplifier = SimplifierIdentity{}) const {
    if(type == OutputType::FLATZINC) {
      print_solution_flatzinc(env, sol, simplifier);
    }
    else {
      print_solution_xml(env, sol, simplifier);
    }
  }
};

} // namespace lala

#endif
