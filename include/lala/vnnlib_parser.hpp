// Copyright 2025 Yi-Nung Tsao

#ifndef LALA_PARSING_VNNLIB_PARSER_HPP
#define LALA_PARSING_VNNLIB_PARSER_HPP

#include <cassert>
#include <cfenv>
#include <cinttypes>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <istream>
#include <set>
#include <streambuf>
#include <string>

#include "battery/shared_ptr.hpp"
#include "lala/logic/ast.hpp"
#include "peglib.h"
#include "solver_output.hpp"

namespace lala {

namespace impl {

template <class Allocator> class VnnlibParser {
  using allocator_type = Allocator;
  using F = TFormula<allocator_type>;
  using SV = peg::SemanticValues;
  using So = Sort<allocator_type>;
  using FSeq = typename F::Sequence;

  bool error;  // If an error was found during parsing.
  bool silent; // If we do not want to output error messages.

public:
  VnnlibParser() : error(false), silent(false) {}

  battery::shared_ptr<F, allocator_type> parse(const std::string &input) {
    peg::parser parser(R"(
                            Statements              <- (DeclareVar / Assertion / Comment)+

                            Integer                 <- < [+-]? [0-9]+ >
                            Real                    <- < (
                                                        'inf'
                                                        / '-inf'
                                                        / [+-]? [0-9]+ (('.' (&'..' / !'.') [0-9]*) / ([Ee][+-]?[0-9]+)) ) >
                            Identifier              <- < [a-zA-Z_][a-zA-Z0-9_]* >

                            BinaryOp                <- '<=' / '>='
                            LogicOp                 <- 'or' / 'and'

                            DeclareVar              <- '(declare-const '  Identifier ' Real' ')'
                            Bound                   <- '(' BinaryOp [ \t]* Identifier [ \t]* (Real / Identifier) [ \t]* ')'
                            Constraint              <- '(' LogicOp [ \t]* Bound* ')' 
                            Assertion               <- '(assert ' Bound ')' / '(assert (' LogicOp+ Constraint* '))' 

                            ~Comment                <- ';' [^\n\r]* [ \n\r\t]*
                            %whitespace             <- [ \n\r\t]*
                           )");

    assert(static_cast<bool>(parser) == true);

    parser["Integer"] = [](const SV &sv) {
      return F::make_z(sv.token_to_number<logic_int>());
    };
    parser["Real"] = [](const SV &sv) {
      return F::make_real(string_to_real(sv.token_to_string()));
    };
    parser["Identifier"] = [](const SV &sv) { return sv.token_to_string(); };
    parser["BinaryOp"] = [](const SV &sv) { return sv.token_to_string(); };
    parser["LogicOp"] = [](const SV &sv) { return sv.token_to_string(); };
    parser["DeclareVar"] = [this](const SV &sv) {
      return make_variable_decl(sv);
    };
    parser["Bound"] = [this](const SV &sv) { return make_bound(sv); };
    parser["Constraint"] = [this](const SV &sv) { return make_constraint(sv); };
    parser["Assertion"] = [this](const SV &sv) { return make_assertion(sv); };

    F f;
    if (parser.parse(input.c_str(), f) && !error) {
      return battery::make_shared<TFormula<Allocator>, Allocator>(std::move(f));
    } else {
      return nullptr;
    }
  }

private:
  static F f(const std::any &any) { return std::any_cast<F>(any); }

  F make_error(const SV &sv, const std::string &msg) {
    if (!silent) {
      std::cerr << sv.line_info().first << ":" << sv.line_info().second << ":"
                << msg << std::endl;
    }
    error = true;

    return F::make_false();
  }

  F make_variable_decl(const SV &sv) {
    auto name = std::any_cast<std::string>(sv[0]);
    auto ty = So(So::Real);

    return F::make_exists(UNTYPED, LVar<allocator_type>(name.data()), ty);
  }

  F make_bound(const SV &sv) {
    auto binary_operator = std::any_cast<std::string>(sv[0]);
    auto name = std::any_cast<std::string>(sv[1]);

    if (binary_operator == "<=") {
      return F::make_binary(f(name), LEQ, f(sv[2]));
    } else if (binary_operator == ">=") {
      return F::make_binary(f(name), GEQ, f(sv[2]));
    }
  }

  F make_constraint(const SV &sv) {
    auto logic_operator = std::any_cast<std::string>(sv[0]);

    // We suppose the logic operator is always "and" for now.
    FSeq seq;
    for (int i = 1; i < sv.size(); ++i) {
      seq.push_back(f(sv[i]));
    }

    return F::make_nary(AND, std::move(seq));
  }

  F make_assertion(const SV &sv) {
    if (sv.size() == 1) {
      return f(sv[0]);
    } else {
      FSeq disjuncts;
      for (int i = 0; i < sv.size(); ++i) {
        disjuncts.push_back(f(sv[i]));
      }

      return F::make_nary(OR, std::move(disjuncts));
    }
  }
};
} // namespace impl
} // namespace lala

#endif
