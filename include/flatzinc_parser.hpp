// Copyright 2022 Pierre Talbot

#ifndef FLATZINC_PARSER_HPP
#define FLATZINC_PARSER_HPP

#include "peglib.h"
#include <cassert>
#include <cstdlib>
#include <string>
#include <istream>
#include <fstream>
#include <streambuf>
#include <iostream>
#include <cfenv>

#include "logic/ast.hpp"
#include "shared_ptr.hpp"

namespace lala {
  namespace impl {
    /** Unfortunately, I'm really not sure this function works in all cases due to compiler bugs with rounding modes... */
    inline logic_real string_to_real(const std::string& s) {
      #ifndef __GNUC__
        #pragma STDC FENV_ACCESS ON
      #endif
      int old = std::fegetround();
      int r = std::fesetround(FE_DOWNWARD);
      assert(r == 0);
      double lb = std::strtod(s.c_str(), nullptr);
      r = std::fesetround(FE_UPWARD);
      assert(r == 0);
      double ub = std::strtod(s.c_str(), nullptr);
      std::fesetround(old);
      return battery::make_tuple(lb, ub);
    }
  }

  template<class Allocator>
  battery::shared_ptr<TFormula<Allocator>, Allocator> parse_flatzinc_str(const std::string& input) {

    using F = TFormula<Allocator>;

    peg::parser parser(R"(
        Statements  <- VariableDecl+

        Literal     <- Boolean / Real / Integer / Identifier

        Identifier  <- < [a-zA-Z_][a-zA-Z0-9_]* >
        Boolean     <- < 'true' / 'false' >
        Real        <- < (
             'inf'
           / '-inf'
           / [+-]? [0-9]+ (('.' [0-9]*) / ([Ee][+-]?[0-9]+)) ) >
        Integer     <- < [+-]? [0-9]+ >

        VariableDecl <- 'var' Type ':' Identifier Annotations ';'

        IntType <- 'int'
        FloatType <- 'float'
        BoolType <- 'bool'
        SetType <- 'set' 'of' Type
        Type <- IntType / FloatType / BoolType / SetType

        Annotations <- ('::' Identifier ('(' Literal ')')?)*

        %whitespace <- [ \n\r\t]*
    )");

    assert(static_cast<bool>(parser) == true);

    parser["Integer"] = [](const peg::SemanticValues &vs) {
      return F::make_z(vs.token_to_number<logic_int>());
    };

    parser["Real"] = [](const peg::SemanticValues &vs) {
      return F::make_real(impl::string_to_real(vs.token_to_string()));
    };

    parser["Boolean"] = [](const peg::SemanticValues &vs) {
      return F::make_z(vs.token_to_string() == "true" ? 1 : 0);
    };

    parser["Identifier"] = [](const peg::SemanticValues &vs) {
      return LVar<Allocator>(vs.token_to_string().c_str());
    };

    parser["IntType"] = [](const peg::SemanticValues &vs) {
      return CType<Allocator>(CType<Allocator>::Int);
    };

    parser["FloatType"] = [](const peg::SemanticValues &vs) {
      return CType<Allocator>(CType<Allocator>::Real);
    };

    parser["BoolType"] = [](const peg::SemanticValues &vs) {
      return CType<Allocator>(CType<Allocator>::Int);
    };

    parser["SetType"] = [](const peg::SemanticValues &vs) {
      CType<Allocator> sub_ty = std::any_cast<CType<Allocator>>(vs[0]);
      return CType<Allocator>(CType<Allocator>::Set, std::move(sub_ty));
    };

    parser["Annotations"] = [](const peg::SemanticValues &vs) {
      return vs;
    };

    parser["VariableDecl"] = [](const peg::SemanticValues &vs) {
      auto ty = std::any_cast<CType<Allocator>>(vs[0]);
      auto f = F::make_exists(UNTYPED,
        std::any_cast<LVar<Allocator>>(vs[1]),
        ty,
        ty.default_approx());
      auto annots = std::any_cast<peg::SemanticValues>(vs[2]);
      for(int i = 0; i < annots.size(); ++i) {
        auto name = std::any_cast<LVar<Allocator>>(annots[i]);
        if(name == "abstract") {
          ++i;
          AType ty = std::any_cast<F>(annots[i]).z();
          f.type_as(ty);
        }
        else if(name == "under") { f.approx_as(UNDER); }
        else if(name == "exact") { f.approx_as(EXACT); }
        else if(name == "over") { f.approx_as(OVER); }
        else if(name == "is_defined_var") {}
        else if(name == "var_is_introduced") {}
        else if(name == "output_var") {}
        else {
          std::cerr << "Annotation " << name.data() << " is unknown." << std::endl;
        }
      }
      return f;
    };

    parser["Statements"] = [](const peg::SemanticValues &vs) {
      if(vs.size() == 1) return std::any_cast<F>(vs[0]);
      else {
        typename F::Sequence children;
        children.reserve(vs.size());
        for(int i = 0; i < vs.size(); ++i) {
          children.push_back(std::any_cast<F>(vs[i]));
        }
        return F::make_nary(AND, std::move(children));
      }
    };

    F f;
    parser.parse(input.c_str(), f);
    return battery::make_shared<TFormula<Allocator>, Allocator>(std::move(f));
  }

  template<class Allocator>
  battery::shared_ptr<TFormula<Allocator>, Allocator> parse_flatzinc(const std::string& filename) {
    std::ifstream t(filename);
    std::string input((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    return parse_flatzinc_str<Allocator>(input);
  }
}

#endif
