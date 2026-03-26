// Copyright 2025 Yi-Nung Tsao

#ifndef LALA_PARSING_SMT_PARSER_HPP
#define LALA_PARSING_SMT_PARSER_HPP

#include "peglib.h"
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
#include "flatzinc_parser.hpp"

namespace lala {

namespace impl {

template <class Allocator>
class SMTParser {
  using allocator_type = Allocator;
  using F = TFormula<allocator_type>;
  using SV = peg::SemanticValues;
  using So = Sort<allocator_type>;
  using FSeq = typename F::Sequence;

  bool error;   // If an error was found during parsing.
  bool silent;  // If we do not want to output error messages.

 public:
  SMTParser() : error(false), silent(false) {}

  F parse(const std::string& input) {
			peg::parser parser(R"(
				Statements    <- (DeclareVar / DeclareFun / Assertion / Comment)+

				Integer       <- < [+-]?[0-9]+ >
				Real          <- < ('inf' / '-inf' /
														[+-]?[0-9]+ (('.' (&'..' / !'.') [0-9]*) /
														([Ee][+-]?[0-9]+)) ) >
				Identifier    <- < [a-zA-Z_][a-zA-Z0-9_]* >

				BinaryOp      <- < '<=' / '>=' / '=' / '>' / '<' >
				LogicOp       <- < 'and' / 'or' >
				ArithOp       <- < '+' / '-' / '*' / '/' >

				Term          <- Neg / Arith / Real / Integer / Identifier
				Neg           <- '(' '-' Term ')'
				Arith         <- '(' ArithOp Term Term ')'

				DeclareVar    <- '(' 'declare-const' Identifier 'Real' ')'
				DeclareFun    <- '(' 'declare-fun' Identifier '(' ')' 'Real' ')'
        Formula       <- Bound / Constraint
        Bound         <- '(' BinaryOp Term Term ')'
        Constraint    <- '(' LogicOp Formula+ ')'
        Assertion     <- '(' 'assert' Formula ')'

				~Comment       <- ';' [^\n\r]* [ \n\r\t]*
				%whitespace   <- [ \n\r\t]*
			)");
    assert(static_cast<bool>(parser) == true);

    parser["Statements"] = [this](const SV& sv) { return make_statements(sv); };
    parser["Integer"] = [](const SV& sv) { return F::make_z(sv.token_to_number<logic_int>()); };
    parser["Real"] = [](const SV& sv) { return F::make_real(impl::string_to_real(sv.token_to_string())); };
    parser["Identifier"] = [](const SV& sv) { return sv.token_to_string(); };
    parser["BinaryOp"] = [](const SV& sv) { return sv.token_to_string(); };
    parser["LogicOp"] = [](const SV& sv) { return sv.token_to_string(); };
    parser["ArithOp"] = [](const SV& sv) { return sv.token_to_string(); };
    parser["DeclareVar"] = [this](const SV& sv) { return make_variable_decl(sv); };
    parser["DeclareFun"] = [this](const SV& sv) { return make_variable_decl(sv); };
    parser["Neg"] = [this](const SV& sv) { return make_neg(sv); };
    parser["Arith"] = [this](const SV& sv) { return make_arith(sv); };
    parser["Bound"] = [this](const SV& sv) { return make_bound(sv); };
    parser["Formula"] = [this](const SV& sv) { return f(sv[0]); };
    parser["Constraint"] = [this](const SV& sv) { return make_constraint(sv); };
    parser["Assertion"] = [this](const SV& sv) { return make_assertion(sv); };

    F smt_formulas;
    if (parser.parse(input.c_str(), smt_formulas) && !error) {
      return smt_formulas; 
    } 
    else {
      std::cerr << "SMT parsing is failed." << std::endl;
      return F::make_false();
    }
  }

 private:
  static F f(const std::any& any) { return std::any_cast<F>(any); }

  F make_error(const SV& sv, const std::string& msg) {
    if (!silent) {
      std::cerr << sv.line_info().first << ":" << sv.line_info().second << ":"
                << msg << std::endl;
    }
    error = true;

    return F::make_false();
  }

  F make_statements(const SV& sv) {
    if (sv.size() == 1) {
      return f(sv[0]);
    } 
    else {
      FSeq children;
      for (size_t i = 0; i < sv.size(); ++i) {
        F formula = f(sv[i]);
        if (!formula.is_true()) {
          children.push_back(formula);
        }
      }
      return F::make_nary(AND, std::move(children));
    }
  }

  F make_variable_decl(const SV& sv) { 
    // For nnv project, all the variables wil be created in onnx_parser.hpp.
    return F::make_true(); 
  }

  F any_to_termF(const std::any& any) {
    try {
      return f(any);
    } catch (const std::bad_any_cast&) {
      auto name = std::any_cast<std::string>(any);
      return F::make_lvar(UNTYPED, LVar<allocator_type>(name.data()));
    }
  }

  F make_arith(const SV& sv) {
    auto arith_operator = std::any_cast<std::string>(sv[0]);
    Sig sig;
    if(arith_operator == "+") sig = ADD;
    else if(arith_operator == "-") sig = SUB;
    else if(arith_operator == "*") sig = MUL;
    else {
      assert(arith_operator == "/");
      sig = DIV;
    }
    return F::make_binary(any_to_termF(sv[1]), sig, any_to_termF(sv[2]));
  }

  F make_neg(const SV& sv) {
    return F::make_unary(NEG, any_to_termF(sv[0]));
  }

  F make_bound(const SV& sv) {
    auto binary_operator = std::any_cast<std::string>(sv[0]);
    Sig sig;
    if (binary_operator == "=") sig = EQ;
    else if (binary_operator == "<=") sig = LEQ;
    else if (binary_operator == ">=") sig = GEQ;
    else if (binary_operator == ">") sig = GT;
    else {
      assert(binary_operator == "<");
      sig = LT;
    }

    return F::make_binary(any_to_termF(sv[1]), sig, any_to_termF(sv[2]));
  }

  F make_constraint(const SV& sv) {
    auto logic_operator = std::any_cast<std::string>(sv[0]);
    FSeq seq;
    for (int i = 1; i < sv.size(); ++i) {
      seq.push_back(f(sv[i]));
    }

    if (logic_operator == "and") {
      return F::make_nary(AND, std::move(seq));
    } else {
      return F::make_nary(OR, std::move(seq));
    }
  }

  F make_assertion(const SV& sv) {
    if (sv.size() == 1) {
      return f(sv[0]);
    } 
    else {
      FSeq disjuncts;
      auto logic_operator = std::any_cast<std::string>(sv[0]); // OR
      for (int i = 1; i < sv.size(); ++i) {
        disjuncts.push_back(f(sv[i]));
      }

      return F::make_nary(OR, std::move(disjuncts));
    }
  }
};
}  // namespace impl

template <class Allocator>
TFormula<Allocator> parse_smt_str(const std::string& input) {
  impl::SMTParser<Allocator> parser;
  return parser.parse(input);
}

template <class Allocator>
TFormula<Allocator> parse_smt(const std::string& filename) {
  std::ifstream t(filename);
  if (t.is_open()) {
    std::string input((std::istreambuf_iterator<char>(t)),
                      std::istreambuf_iterator<char>());
    return parse_smt_str<Allocator>(input);
  } else {
    std::cerr << "File `" << filename << "` does not exists." << std::endl;
  }
  return TFormula<Allocator>::make_false();
}

}  // namespace lala

#endif
