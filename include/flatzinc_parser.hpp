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

template<class Allocator>
class FlatZincOutput {
  using bstring = battery::string<Allocator>;
  template<class T> using bvector = battery::vector<T, Allocator>;
  using array_dim_t = bvector<battery::tuple<int,int>>;
  using F = TFormula<Allocator>;

  bvector<bstring> output_vars;
  // For each array, we store its output dimension characteristics and the list of the variables in the array.
  bvector<battery::tuple<bstring, array_dim_t, bvector<bstring>>> output_arrays;

public:
  CUDA FlatZincOutput() = default;
  CUDA FlatZincOutput(FlatZincOutput&&) = default;

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
      output_arrays.push_back(battery::make_tuple<bstring, array_dim_t, bvector<bstring>>(bstring(array_name), {}, {}));
      idx = output_arrays.size() - 1;
      // Add the dimension of the array.
      for(int i = 0; i < sv.size(); ++i) {
        auto range = std::any_cast<F>(sv[i]);
        for(int j = 0; j < range.s().size(); ++j) {
          const auto& itv = range.s()[j];
          battery::get<1>(output_arrays[idx]).push_back(battery::make_tuple(battery::get<0>(itv).z(), battery::get<1>(itv).z()));
        }
      }
    }
    battery::get<2>(output_arrays[idx]).push_back(var_name);
  }

  void add_var(const bstring& var_name) {
    output_vars.push_back(var_name);
  }

  template <class Env, class A>
  CUDA void print_solution(const Env& env, const A& sol) const {
    for(int i = 0; i < output_vars.size(); ++i) {
      printf("%s=", output_vars[i].data());
      AVar avar = env.variable_of(output_vars[i]).avars[0];
      sol.project(avar).lb().print();
      printf(";\n");
    }
    for(int i = 0; i < output_arrays.size(); ++i) {
      const auto& dims = battery::get<1>(output_arrays[i]);
      const auto& array_vars = battery::get<2>(output_arrays[i]);
      printf("%s=array%dd(", battery::get<0>(output_arrays[i]).data(), dims.size());
      for(int j = 0; j < dims.size(); ++j) {
        printf("%d..%d,", battery::get<0>(dims[j]), battery::get<1>(dims[j]));
      }
      printf("[");
      for(int j = 0; j < array_vars.size(); ++j) {
        AVar avar = env.variable_of(array_vars[j]).avars[0];
        sol.project(avar).lb().print();
        if(j+1 != array_vars.size()) {
          printf(",");
        }
      }
      printf("]);\n");
    }
  }
};


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

template <class Allocator>
class FlatZincParser {
  using allocator_type = Allocator;
  using F = TFormula<allocator_type>;
  using SV = peg::SemanticValues;
  using Set = logic_set<F, allocator_type>;
  using So = Sort<allocator_type>;

  std::map<std::string, F> params; // Name and value of the parameters occuring in the model.
  std::map<std::string, int> arrays; // Size of all named arrays (parameters and variables).
  bool error; // If an error was found during parsing.
  bool silent; // If we do not want to output error messages.
  FlatZincOutput<Allocator>& output;

public:
  FlatZincParser(FlatZincOutput<Allocator>& output): error(false), silent(false), output(output) {}

  battery::shared_ptr<F, allocator_type> parse(const std::string& input) {
    peg::parser parser(R"(
      Statements  <- (VariableDecl / VarArrayDecl / ParameterDecl / ConstraintDecl / SolveItem / Comment)+

      Literal      <- Boolean / Real / Integer / ArrayAccess / VariableLit / Set
      RangeLiteral <- SetRange / Literal
      InnerRangeLiteral <- InnerSetRange / Literal

      VariableLit <- Identifier
      Identifier  <- < [a-zA-Z_][a-zA-Z0-9_]* >
      Boolean     <- < 'true' / 'false' >
      Real        <- < (
           'inf'
         / '-inf'
         / [+-]? [0-9]+ (('.' (&'..' / !'.') [0-9]*) / ([Ee][+-]?[0-9]+)) ) >
      Integer     <- < [+-]? [0-9]+ >
      Set         <- '{' '}' / '{' InnerRangeLiteral (',' InnerRangeLiteral)* '}'
      InnerSetRange    <- Literal '..' Literal
      SetRange <- InnerSetRange
      ArrayAccess <- Identifier '[' (VariableLit / Integer) ']'

      VariableDecl <- 'var' ValueType ':' Identifier Annotations ('=' Literal)? ';'
      VarArrayDecl <- 'array' '[' IndexSet ']' 'of' 'var' ValueType ':' Identifier Annotations ('=' LiteralArray)? ';'

      SetValue <- 'set' 'of' (SetRange / Set)

      ValueType <- Type
                / SetValue
                / SetRange
                / Set

      IntType <- 'int'
      RealType <- 'float' / 'real'
      BoolType <- 'bool'
      SetType <- 'set' 'of' Type
      Type <- IntType / RealType / BoolType / SetType

      Annotation <- Identifier ('(' Parameter (',' Parameter)* ')')?
      Annotations <- ('::'  Annotation)*

      ConstraintDecl <- 'constraint' (PredicateCall / Boolean) Annotations ';'

      LiteralInExpression <- RangeLiteral !'('
      FunctionCall <-  Identifier '(' Parameter (',' Parameter )* ')'
      Parameter <- LiteralInExpression / FunctionCall / LiteralArray
      PredicateCall <- Identifier '(' Parameter (',' Parameter)* ')'

      MinimizeItem <- 'minimize' RangeLiteral
      MaximizeItem <- 'maximize' RangeLiteral
      SatisfyItem <- 'satisfy'
      SolveItem <- 'solve' (MinimizeItem / MaximizeItem / SatisfyItem) ';'

      LiteralArray <- '[' RangeLiteral (',' RangeLiteral)* ']'
      ParameterExpr <- RangeLiteral / LiteralArray
      IndexSet <- '1' '..' Integer
      ArrayType <- 'array' '[' IndexSet ']' 'of' Type
      ParameterType <- Type / ArrayType
      ParameterDecl <- ParameterType ':' Identifier '=' ParameterExpr ';'

      ~Comment    <- '%' [^\n\r]* [ \n\r\t]*
      %whitespace <- [ \n\r\t]*
    )");

    assert(static_cast<bool>(parser) == true);

    parser["Integer"] = [](const SV &sv) { return F::make_z(sv.token_to_number<logic_int>()); };
    parser["Real"] = [](const SV &sv) { return F::make_real(string_to_real(sv.token_to_string())); };
    parser["Boolean"] = [](const SV &sv) { return sv.token_to_string() == "true" ? F::make_true() : F::make_false(); };
    parser["Identifier"] = [](const SV &sv) { return sv.token_to_string(); };
    parser["Set"] = [this](const SV &sv) { return make_set_literal(sv); };
    parser["InnerSetRange"] = [this](const SV &sv) { return battery::make_tuple(f(sv[0]), f(sv[1])); };
    parser["SetRange"] = [this](const SV &sv) { return F::make_set(Set({itv(sv[0])})); };
    parser["ArrayAccess"] = [this](const SV &sv) { return make_access_literal(sv); };
    parser["LiteralArray"] = [](const SV &sv) { return sv; };
    parser["ParameterDecl"] = [this] (const SV &sv) { return make_parameter_decl(sv); };
    parser["VariableLit"] = [](const SV &sv) { return F::make_lvar(UNTYPED, LVar<Allocator>(std::any_cast<std::string>(sv[0]))); };
    parser["IntType"] = [](const SV &sv) { return So(So::Int); };
    parser["RealType"] = [](const SV &sv) { return So(So::Real); };
    parser["BoolType"] = [](const SV &sv) { return So(So::Bool); };
    parser["SetType"] = [](const SV &sv) { return So(So::Set, std::any_cast<Sort<Allocator>>(sv[0])); };
    parser["Annotations"] = [](const SV &sv) { return sv; };
    parser["Annotation"] = [](const SV &sv) { return sv; };
    // When we have `var set of 1..5: s;`, what it means is that `s in {}..{1..5}`, i.e., a set between {} and {1..5}.
    parser["SetValue"] = [](const SV &sv) { return F::make_set(Set({battery::make_tuple(F::make_set(Set{}), f(sv[0]))})); };
    parser["VariableDecl"] = [this](const SV &sv) { return make_variable_init_decl(sv); };
    parser["VarArrayDecl"] = [this](const SV &sv) { return make_variable_array_decl(sv); };
    parser["ConstraintDecl"] = [this](const SV &sv) { return update_with_annotations(sv, f(sv[0]), std::any_cast<SV>(sv[1])); };
    parser["FunctionCall"] = [this](const SV &sv) { return function_call(sv); };
    parser["PredicateCall"] = [this](const SV &sv) { return predicate_call(sv); };
    parser["Statements"] = [this](const SV &sv) { return make_statements(sv); };
    parser["MinimizeItem"] = [](const SV &sv) { return F::make_unary(MINIMIZE, f(sv[0])); };
    parser["MaximizeItem"] = [](const SV &sv) { return F::make_unary(MAXIMIZE, f(sv[0])); };
    parser["SatisfyItem"] = [](const SV &sv) { return F(); };
    parser.set_logger([](size_t line, size_t col, const std::string& msg, const std::string &rule) {
      std::cerr << line << ":" << col << ": " << msg << "\n";
    });

    F f;
    if(parser.parse(input.c_str(), f) && !error) {
      // for(const auto& paramValue : params) {
      //   std::cout << paramValue.first << " = ";
      //   paramValue.second.print(false);
      //   std::cout << std::endl;
      // }
      return battery::make_shared<TFormula<Allocator>, Allocator>(std::move(f));
    }
    else {
      return nullptr;
    }
  }

  private:
    static F f(const std::any& any) {
      return std::any_cast<F>(any);
    }

    static battery::tuple<F, F> itv(const std::any& any) {
      return std::any_cast<battery::tuple<F, F>>(any);
    }

    F make_error(const SV& sv, const std::string& msg) {
      if(!silent) {
        std::cerr << sv.line_info().first << ":" << sv.line_info().second << ":" << msg << std::endl;
      }
      error = true;
      return F::make_false();
    }

    F make_arity_error(const SV& sv, Sig sig, int expected, int obtained) {
      return make_error(sv, "The symbol `" + std::string(string_of_sig(sig)) +
        "` expects `" + std::to_string(expected) + " parameters" +
        ", but we got `" + std::to_string(obtained) + "` parameters.");
    }

    F make_set_literal(const SV& sv) {
      logic_set<F, allocator_type> set;
      for(int i = 0; i < sv.size(); ++i) {
        try {
          auto range = std::any_cast<battery::tuple<F,F>>(sv[i]);
          set.push_back(range);
        }
        catch(std::bad_any_cast) {
          bool element_added = false;
          auto element = f(sv[i]);
          if(element.is(F::Z) && set.size() > 0) {
            auto& ub = battery::get<1>(set.back());
            if(!ub.is(F::Z)) {
              return make_error(sv, "Elements in a set are expected to be all of the same type.");
            }
            else if(ub.z() == element.z() - 1) {
              ub.z() = element.z();
              element_added = true;
            }
          }
          else if(element.is(F::LV)) {
            std::string name(element.lv().data());
            if(params.contains(name)) {
              set.push_back(std::make_tuple(params[name], params[name]));
              element_added = true;
            }
            else {
              return make_error(sv, "Undeclared parameter `" + name + "`.");
            }
          }
          if(!element_added) {
            set.push_back(std::make_tuple(element, element));
          }
        }
      }
      return F::make_set(set);
    }

    F make_access_literal(const SV& sv) {
      auto name = std::any_cast<std::string>(sv[0]);
      auto index = f(sv[1]);
      int idx = -1;
      if(index.is(F::Z)) {
        idx = index.z();
      }
      else if(index.is(F::LV)) {
        if(params.contains(index.lv().data())) {
          auto pindex = params[index.lv().data()];
          if(pindex.is(F::Z)) {
            idx = pindex.z();
          }
        }
      }
      if(idx == -1) {
        return make_error(sv, "Given `a[b]`, `b` must be an integer or an integer parameter.");
      }
      auto access_var = make_array_access(name, idx-1);
      if(params.contains(access_var)) {
        return params[access_var];
      }
      else {
        return F::make_lvar(UNTYPED, access_var);
      }
    }

    F make_parameter_decl(const SV& sv) {
      std::string identifier(std::any_cast<std::string>(sv[1]));
      try {
        if(params.contains(identifier)) {
          return make_error(sv, ("Parameter `" + identifier + "` already declared.").c_str());
        }
        else {
          params[identifier] = f(sv[2]);
        }
      } catch(std::bad_any_cast) {
        auto array = std::any_cast<SV>(sv[2]);
        arrays[identifier] = array.size();
        for(int i = 0; i < array.size(); ++i) {
          auto id = make_array_access(identifier, i);
          params[id] = f(array[i]);
        }
      }
      return F::make_true();
    }

    F make_variable_array_decl(const SV& sv) {
      int arraySize = f(sv[0]).z();
      auto name = std::any_cast<std::string>(sv[2]);
      arrays[name] = arraySize;
      battery::vector<F, Allocator> decl;
      for(int i = 0; i < arraySize; ++i) {
        decl.push_back(make_variable_decl(sv, make_array_access(name, i), sv[1], sv[3]));
      }
      if(sv.size() == 5) {
        auto array = resolve_array(sv, sv[4]);
        if(!array.is(F::Seq)) {
          return array;
        }
        for(int i = 0; i < array.seq().size(); ++i) {
          decl.push_back(F::make_binary(
            F::make_lvar(UNTYPED, LVar<Allocator>(make_array_access(name, i))),
            EQ,
            array.seq(i)));
        }
      }
      return F::make_nary(AND, std::move(decl));
    }

    F update_with_annotations(const SV& sv, F formula, const SV& annots) {
      for(int i = 0; i < annots.size(); ++i) {
        auto annot = std::any_cast<SV>(annots[i]);
        auto name = std::any_cast<std::string>(annot[0]);
        if(name == "abstract") {
          AType ty = f(annot[1]).z();
          formula.type_as(ty);
        }
        else if(name == "under") { formula.approx_as(UNDER); }
        else if(name == "exact") { formula.approx_as(EXACT); }
        else if(name == "over") { formula.approx_as(OVER); }
        else if(name == "is_defined_var") {}
        else if(name == "defines_var") {}
        else if(name == "var_is_introduced") {}
        else if(name == "output_var" && formula.is(F::E)) {
          output.add_var(battery::get<0>(formula.exists()));
        }
        else if(name == "output_array" && formula.is(F::E)) {
          auto array_name = std::any_cast<std::string>(sv[2]);
          auto dims = std::any_cast<SV>(annot[1]);
          output.add_array_var(array_name, battery::get<0>(formula.exists()), dims);
        }
        else {
          std::cerr << "Annotation " + name + " is unknown was ignored." << std::endl;
        }
      }
      return std::move(formula);
    }

    F make_binary(Sig sig, const SV &sv) {
      if(sv.size() != 3) {
        return make_arity_error(sv, sig, 2, sv.size() - 1);
      }
      return F::make_binary(f(sv[1]), sig, f(sv[2]));
    }

    F make_unary_fun_eq(Sig sig, const SV &sv, Sig eq_kind = EQ) {
      if(sv.size() != 3) {
        return make_arity_error(sv, sig, 1, sv.size() - 2);
      }
      auto fun = F::make_unary(sig, f(sv[1]));
      return F::make_binary(fun, eq_kind, f(sv[2]));
    }

    F make_unary_fun(Sig sig, const SV &sv) {
      if(sv.size() != 2) {
        return make_arity_error(sv, sig, 1, sv.size() - 1);
      }
      return F::make_unary(sig, f(sv[1]));
    }

    F make_binary_fun_eq(Sig sig, const SV &sv, Sig eq_kind = EQ) {
      if(sv.size() != 4) {
        return make_arity_error(sv, sig, 2, sv.size() - 2);
      }
      auto fun = F::make_binary(f(sv[1]), sig, f(sv[2]));
      return F::make_binary(fun, eq_kind, f(sv[3]));
    }

    F make_binary_fun(Sig sig, const SV &sv) {
      if(sv.size() != 3) {
        return make_arity_error(sv, sig, 2, sv.size() - 1);
      }
      return F::make_binary(f(sv[1]), sig, f(sv[2]));
    }

    F make_float_in(const SV &sv) {
      return F::make_binary(
          F::make_binary(f(sv[1]), GEQ, f(sv[2])),
          AND,
          F::make_binary(f(sv[1]), LEQ, f(sv[3])));
    }

    F make_log(int base, const SV &sv) {
      return F::make_binary(f(sv[1]), LOG, F::make_z(base));
    }

    F make_log_eq(int base, const SV &sv) {
      return F::make_binary(make_log(base, sv), EQ, f(sv[2]));
    }

    F predicate_call(const SV &sv) {
      auto name = std::any_cast<std::string>(sv[0]);
      if(name == "int_le") { return make_binary(LEQ, sv); }
      else if(name == "int_lt") { return make_binary(LT, sv); }
      else if(name == "int_ge") { return make_binary(GEQ, sv); }
      else if(name == "int_gt") { return make_binary(GT, sv); }
      else if(name == "int_eq") { return make_binary(EQ, sv); }
      else if(name == "int_ne") { return make_binary(NEQ, sv); }
      else if(name == "int_abs") { return make_unary_fun_eq(ABS, sv); }
      else if(name == "int_neg") { return make_unary_fun_eq(NEG, sv); }
      else if(name == "int_div") { return make_binary_fun_eq(EDIV, sv); }
      else if(name == "int_mod") { return make_binary_fun_eq(EMOD, sv); }
      else if(name == "int_plus") { return make_binary_fun_eq(ADD, sv); }
      else if(name == "int_minus") { return make_binary_fun_eq(SUB, sv); }
      else if(name == "int_pow") { return make_binary_fun_eq(POW, sv); }
      else if(name == "int_times") { return make_binary_fun_eq(MUL, sv); }
      else if(name == "int_max") { return make_binary_fun_eq(MAX, sv); }
      else if(name == "int_min") { return make_binary_fun_eq(MIN, sv); }
      else if(name == "int_eq_reif") { return make_binary_fun_eq(EQ, sv, EQUIV); }
      else if(name == "int_le_reif") { return make_binary_fun_eq(LEQ, sv, EQUIV); }
      else if(name == "int_lt_reif") { return make_binary_fun_eq(LT, sv, EQUIV); }
      else if(name == "int_ne_reif") { return make_binary_fun_eq(NEQ, sv, EQUIV); }
      else if(name == "bool2int") { return make_binary(EQ, sv); }
      else if(name == "bool_eq") { return make_binary(EQ, sv); }
      else if(name == "bool_le") { return make_binary(LEQ, sv); }
      else if(name == "bool_lt") { return make_binary(LT, sv); }
      else if(name == "bool_eq_reif") { return make_binary_fun_eq(EQ, sv, EQUIV); }
      else if(name == "bool_le_reif") { return make_binary_fun_eq(LEQ, sv, EQUIV); }
      else if(name == "bool_lt_reif") { return make_binary_fun_eq(LT, sv, EQUIV); }
      else if(name == "bool_and") { return make_binary_fun_eq(AND, sv, EQUIV); }
      else if(name == "bool_not") { return make_binary(NOT, sv); }
      else if(name == "bool_or") { return make_binary_fun_eq(OR, sv, EQUIV); }
      else if(name == "bool_xor") {
        if(sv.size() == 3) { return make_binary(XOR, sv); }
        else { return make_binary_fun_eq(XOR, sv, EQUIV); }
      }
      else if(name == "set_card") { return make_unary_fun_eq(CARD, sv); }
      else if(name == "set_diff") { return make_binary_fun_eq(DIFFERENCE, sv); }
      else if(name == "set_eq") { return make_binary(EQ, sv); }
      else if(name == "set_eq_reif") { return make_binary_fun_eq(EQ, sv, EQUIV); }
      else if(name == "set_in") { return make_binary(IN, sv); }
      else if(name == "set_in_reif") { return make_binary_fun_eq(IN, sv, EQUIV); }
      else if(name == "set_intersect") { return make_binary_fun_eq(INTERSECTION, sv, EQUIV); }
      else if(name == "set_union") { return make_binary_fun_eq(UNION, sv, EQUIV); }
      else if(name == "set_ne") { return make_binary(NEQ, sv); }
      else if(name == "set_ne_reif") { return make_binary_fun_eq(NEQ, sv, EQUIV); }
      else if(name == "set_subset") { return make_binary(SUBSETEQ, sv); }
      else if(name == "set_subset_reif") { return make_binary_fun_eq(SUBSETEQ, sv, EQUIV); }
      else if(name == "set_superset") { return make_binary(SUPSETEQ, sv); }
      else if(name == "set_symdiff") { return make_binary_fun_eq(SYMMETRIC_DIFFERENCE, sv, EQUIV); }
      else if(name == "set_le") { return make_binary(LEQ, sv); }
      else if(name == "set_le_reif") { return make_binary_fun_eq(LEQ, sv, EQUIV); }
      else if(name == "set_lt") { return make_binary(LT, sv); }
      else if(name == "set_lt_reif") { return make_binary_fun_eq(LT, sv, EQUIV); }
      else if(name == "float_abs") { return make_binary_fun_eq(ABS, sv); }
      else if(name == "float_neg") { return make_binary_fun_eq(NEG, sv); }
      else if(name == "float_plus") { return make_binary_fun_eq(ADD, sv); }
      else if(name == "float_minus") { return make_binary_fun_eq(SUB, sv); }
      else if(name == "float_times") { return make_binary_fun_eq(MUL, sv); }
      else if(name == "float_acos") { return make_unary_fun_eq(ACOS, sv); }
      else if(name == "float_acosh") { return make_unary_fun_eq(ACOSH, sv); }
      else if(name == "float_asin") { return make_unary_fun_eq(ASIN, sv); }
      else if(name == "float_asinh") { return make_unary_fun_eq(ASINH, sv); }
      else if(name == "float_atan") { return make_unary_fun_eq(ATAN, sv); }
      else if(name == "float_atanh") { return make_unary_fun_eq(ATANH, sv); }
      else if(name == "float_cos") { return make_unary_fun_eq(COS, sv); }
      else if(name == "float_cosh") { return make_unary_fun_eq(COSH, sv); }
      else if(name == "float_sin") { return make_unary_fun_eq(SIN, sv); }
      else if(name == "float_sinh") { return make_unary_fun_eq(SINH, sv); }
      else if(name == "float_tan") { return make_unary_fun_eq(TAN, sv); }
      else if(name == "float_tanh") { return make_unary_fun_eq(TANH, sv); }
      else if(name == "float_div") { return make_binary(DIV, sv); }
      else if(name == "float_eq") { return make_binary(EQ, sv); }
      else if(name == "float_eq_reif") { return make_binary_fun_eq(EQ, sv, EQUIV); }
      else if(name == "float_le") { return make_binary(LEQ, sv); }
      else if(name == "float_le_reif") { return make_binary_fun_eq(LEQ, sv, EQUIV); }
      else if(name == "float_ne") { return make_binary(NEQ, sv); }
      else if(name == "float_ne_reif") { return make_binary_fun_eq(NEQ, sv, EQUIV); }
      else if(name == "float_lt") { return make_binary(LT, sv); }
      else if(name == "float_lt_reif") { return make_binary_fun_eq(LT, sv, EQUIV); }
      else if(name == "float_in") { return make_float_in(sv); }
      else if(name == "float_in_reif") {
        return F::make_binary(make_float_in(sv), EQUIV, f(sv[4]));
      }
      else if(name == "float_log10") { return make_log_eq(10, sv); }
      else if(name == "float_log2") { return make_log_eq(2, sv); }
      else if(name == "float_min") { return make_binary_fun_eq(MIN, sv); }
      else if(name == "float_max") { return make_binary_fun_eq(MAX, sv); }
      else if(name == "float_exp") { return make_unary_fun_eq(EXP, sv); }
      else if(name == "float_ln") { return make_unary_fun_eq(LN, sv); }
      else if(name == "float_pow") { return make_binary_fun_eq(POW, sv); }
      else if(name == "float_sqrt") { return make_unary_fun_eq(SQRT, sv); }
      else if(name == "int2float") { return make_binary(EQ, sv); }
      else if(name == "array_int_element" || name == "array_var_int_element"
        || name == "array_bool_element" || name == "array_var_bool_element"
        || name == "array_set_element" || name == "array_var_set_element"
        || name == "array_float_element" || name == "array_var_float_element")
      {
        return make_element_constraint(name, sv);
      }
      else if(name == "int_lin_eq" || name == "bool_lin_eq" || name == "float_lin_eq" ||
              name == "int_lin_eq_reif" || name == "bool_lin_eq_reif" || name == "float_lin_eq_reif")
      {
        return make_linear_constraint(name, EQ, sv);
      }
      else if(name == "int_lin_le" || name == "bool_lin_le" || name == "float_lin_le" ||
              name == "int_lin_le_reif" || name == "bool_lin_le_reif" || name == "float_lin_le_reif")
      {
        return make_linear_constraint(name, LEQ, sv);
      }
      else if(name == "int_lin_ne" || name == "bool_lin_ne" || name == "float_lin_ne" ||
              name == "int_lin_ne_reif" || name == "bool_lin_ne_reif" || name == "float_lin_ne_reif")
      {
        return make_linear_constraint(name, NEQ, sv);
      }
      else if(name == "array_bool_and") {
        return make_boolean_constraint(name, AND, sv);
      }
      else if(name == "array_bool_or") {
        return make_boolean_constraint(name, OR, sv);
      }
      else if(name == "array_bool_xor") {
        return make_boolean_constraint(name, XOR, sv);
      }
      else if(name == "bool_clause" || name == "bool_clause_reif") {
        return make_boolean_clause(name, sv);
      }
      return make_error(sv, "Unknown predicate `" + name + "`");
    }

    F function_call(const SV &sv) {
      auto name = std::any_cast<std::string>(sv[0]);
      silent = true;
      bool err = error;
      error = false;
      auto p = predicate_call(sv);
      silent = false;
      if(!error) {
        return f(p);
      }
      else {
        error = err;
        if(name == "int_abs") { return make_unary_fun(ABS, sv); }
        else if(name == "int_neg") { return make_unary_fun(NEG, sv); }
        else if(name == "int_div") { return make_binary_fun(EDIV, sv); }
        else if(name == "int_mod") { return make_binary_fun(EMOD, sv); }
        else if(name == "int_plus") { return make_binary_fun(ADD, sv); }
        else if(name == "int_minus") { return make_binary_fun(SUB, sv); }
        else if(name == "int_pow") { return make_binary_fun(POW, sv); }
        else if(name == "int_times") { return make_binary_fun(MUL, sv); }
        else if(name == "int_max") { return make_binary_fun(MAX, sv); }
        else if(name == "int_min") { return make_binary_fun(MIN, sv); }
        else if(name == "bool_and") { return make_binary_fun(AND, sv); }
        else if(name == "bool_not") { return make_unary_fun(NOT, sv); }
        else if(name == "bool_or") { return make_binary_fun(OR, sv); }
        else if(name == "bool_xor") { return make_binary_fun(XOR, sv); }
        else if(name == "set_card") { return make_unary_fun(CARD, sv); }
        else if(name == "set_diff") { return make_binary_fun(DIFFERENCE, sv); }
        else if(name == "set_intersect") { return make_binary_fun(INTERSECTION, sv); }
        else if(name == "set_union") { return make_binary_fun(UNION, sv); }
        else if(name == "set_symdiff") { return make_binary_fun(SYMMETRIC_DIFFERENCE, sv); }
        else if(name == "float_abs") { return make_binary_fun(ABS, sv); }
        else if(name == "float_neg") { return make_binary_fun(NEG, sv); }
        else if(name == "float_plus") { return make_binary_fun(ADD, sv); }
        else if(name == "float_minus") { return make_binary_fun(SUB, sv); }
        else if(name == "float_times") { return make_binary_fun(MUL, sv); }
        else if(name == "float_acos") { return make_unary_fun(ACOS, sv); }
        else if(name == "float_acosh") { return make_unary_fun(ACOSH, sv); }
        else if(name == "float_asin") { return make_unary_fun(ASIN, sv); }
        else if(name == "float_asinh") { return make_unary_fun(ASINH, sv); }
        else if(name == "float_atan") { return make_unary_fun(ATAN, sv); }
        else if(name == "float_atanh") { return make_unary_fun(ATANH, sv); }
        else if(name == "float_cos") { return make_unary_fun(COS, sv); }
        else if(name == "float_cosh") { return make_unary_fun(COSH, sv); }
        else if(name == "float_sin") { return make_unary_fun(SIN, sv); }
        else if(name == "float_sinh") { return make_unary_fun(SINH, sv); }
        else if(name == "float_tan") { return make_unary_fun(TAN, sv); }
        else if(name == "float_tanh") { return make_unary_fun(TANH, sv); }
        else if(name == "float_div") { return make_binary_fun(DIV, sv); }
        else if(name == "float_log10") { return make_log(10, sv); }
        else if(name == "float_log2") { return make_log(2, sv); }
        else if(name == "float_min") { return make_binary_fun(MIN, sv); }
        else if(name == "float_max") { return make_binary_fun(MAX, sv); }
        else if(name == "float_exp") { return make_unary_fun(EXP, sv); }
        else if(name == "float_ln") { return make_unary_fun(LN, sv); }
        else if(name == "float_pow") { return make_binary_fun(POW, sv); }
        else if(name == "float_sqrt") { return make_unary_fun(SQRT, sv); }
        return make_error(sv, "Unknown function or predicate symbol `" + name + "`");
      }
    }

    F make_statements(const SV& sv) {
      if(sv.size() == 1) {
        return f(sv[0]);
      }
      else {
        typename F::Sequence children;
        for(int i = 0; i < sv.size(); ++i) {
          F formula = f(sv[i]);
          if(!formula.is_true()) {
            children.push_back(formula);
          }
        }
        return F::make_nary(AND, std::move(children));
      }
    }

    F make_existential(const SV& sv, const So& ty, const std::string& name, const std::any& sv_annots) {
      auto f = F::make_exists(UNTYPED,
        LVar<allocator_type>(name.data()),
        ty,
        ty.default_approx());
      auto annots = std::any_cast<SV>(sv_annots);
      return update_with_annotations(sv, f, annots);
    }

    template<class S>
    std::string make_array_access(const S& name, int i) {
      return std::string(name.data()) + "[" + std::to_string(i+1) + "]"; // FlatZinc array starts at 1.
    }

    F make_variable_init_decl(const SV& sv) {
      auto name = std::any_cast<std::string>(sv[1]);
      auto var_decl = make_variable_decl(sv, name, sv[0], sv[2]);
      if(sv.size() == 4) {
        return F::make_binary(std::move(var_decl), AND,
          F::make_binary(
            F::make_lvar(UNTYPED, LVar<allocator_type>(name.data())),
            EQ,
            f(sv[3])));
      }
      else {
        return std::move(var_decl);
      }
    }

    F make_variable_decl(const SV& sv, const std::string& name, const std::any& typeVar, const std::any& annots) {
      try {
        auto ty = std::any_cast<So>(typeVar);
        return make_existential(sv, ty, name, annots);
      }
      catch(std::bad_any_cast) {
        auto typeValue = f(typeVar);
        auto inConstraint = F::make_binary(F::make_lvar(UNTYPED, LVar<allocator_type>(name.data())), IN, typeValue);
        auto sort = typeValue.sort();
        if(!sort.has_value() || !sort->is_set()) {
          return make_error(sv, "We only allow type-value of variables to be of type Set.");
        }
        auto exists = make_existential(sv, *(sort->sub), name, annots);
        return F::make_binary(std::move(exists), AND, std::move(inConstraint));
      }
    }

    F make_element_constraint(const std::string& name, const SV& sv) {
      if(sv.size() < 4) {
        return make_error(sv, "`" + name + "` expects 3 parameters, but we got `" + std::to_string(sv.size()-1) + "` parameters");
      }
      auto index = f(sv[1]);
      auto array = resolve_array(sv, sv[2]);
      if(!array.is(F::Seq)) {
        return array;
      }
      auto value = f(sv[3]);
      typename F::Sequence seq;
      for(int i = 0; i < array.seq().size(); ++i) {
        // index = (i+1) ==> varName = value
        seq.push_back(F::make_binary(
          F::make_binary(index, EQ, F::make_z(i+1)),
          IMPLY,
          F::make_binary(array.seq(i), EQ, value)));
      }
      return F::make_nary(AND, std::move(seq));
    }

    // We return the elements inside the array in the form `arr[1] /\ arr[2] /\ ... /\ arr[N]`.
    // The array can either be a literal array directly, or the name of an array.
    F resolve_array(const SV& sv, const std::any& any) {
      typename F::Sequence seq;
      try {
        auto arrayVar = f(any);
        if(arrayVar.is(F::LV)) {
          std::string arrayName(arrayVar.lv().data());
          if(arrays.contains(arrayName)) {
            int size = arrays[arrayName];
            for(int i = 0; i < size; ++i) {
              auto varName = make_array_access(arrayName, i);
              if(params.contains(varName)) {
                seq.push_back(params[varName]);
              }
              else {
                seq.push_back(F::make_lvar(UNTYPED, LVar<allocator_type>(varName.data())));
              }
            }
          }
          else {
            return make_error(sv, "Unknown array parameter `" + arrayName + "`");
          }
        }
        else {
          return make_error(sv, "Expects an array or the name of an array.");
        }
      }
      catch(std::bad_any_cast) {
        auto array = std::any_cast<SV>(any);
        for(int i = 0; i < array.size(); ++i) {
          seq.push_back(f(array[i]));
        }
      }
      return F::make_nary(AND, std::move(seq));
    }

    F make_linear_constraint(const std::string& name, Sig sig, const SV& sv) {
      if(sv.size() != 4 && sv.size() != 5) {
        return make_error(sv, "`" + name + "` expects 3 (or 4 if reified) parameters, but we got `" + std::to_string(sv.size() - 1) + "` parameters");
      }
      auto as = resolve_array(sv, sv[1]);
      if(!as.is(F::Seq)) { return as; }
      auto bs = resolve_array(sv, sv[2]);
      if(!bs.is(F::Seq)) { return bs; }
      auto c = f(sv[3]);
      if(as.seq().size() != bs.seq().size()) {
        return make_error(sv, "`" + name + "` expects arrays of the same size.");
      }
      typename F::Sequence sum;
      for(int i = 0; i < as.seq().size(); ++i) {
        sum.push_back(F::make_binary(as.seq(i), MUL, bs.seq(i)));
      }
      auto linearCons = F::make_binary(F::make_nary(ADD, std::move(sum)), sig, c);
      if(sv.size() == 5) { // reified version.
        return F::make_binary(f(sv[4]), EQUIV, std::move(linearCons));
      }
      else {
        return std::move(linearCons);
      }
    }

    F make_boolean_constraint(const std::string& name, Sig sig, const SV& sv) {
      if(sv.size() != 2 && sv.size() != 3) {
        return make_error(sv, "`" + name + "` expects 1 (or 2 if reified) parameters, but we got `" + std::to_string(sv.size() - 1) + "` parameters");
      }
      auto array = resolve_array(sv, sv[1]);
      if(!array.is(F::Seq)) { return array; }
      if(sv.size() == 3) { // reified
        return F::make_binary(f(sv[2]), EQUIV, F::make_nary(sig, array.seq()));
      }
      else {
        return F::make_nary(sig, array.seq());
      }
    }

    F make_boolean_clause(const std::string& name, const SV& sv) {
      if(sv.size() != 3 && sv.size() != 4) {
        return make_error(sv, "`" + name + "` expects 2 (or 3 if reified) parameters, but we got `" + std::to_string(sv.size() - 1) + "` parameters");
      }
      auto as = resolve_array(sv, sv[1]);
      if(!as.is(F::Seq)) { return as; }
      auto bs = resolve_array(sv, sv[2]);
      if(!bs.is(F::Seq)) { return bs; }
      typename F::Sequence negs;
      for(int i = 0; i < bs.seq().size(); ++i) {
        negs.push_back(F::make_unary(NEG, bs.seq(i)));
      }
      F clause = F::make_binary(F::make_nary(OR, as.seq()), OR, F::make_nary(OR, negs));
      if(sv.size() == 4) {
        return F::make_binary(f(sv[3]), EQUIV, std::move(clause));
      }
      else {
        return std::move(clause);
      }
    }
  };
}

  /** We parse the constraint language FlatZinc as described in the documentation: https://www.minizinc.org/doc-2.4.1/en/fzn-spec.html#specification-of-flatzinc.
   * We also extend FlatZinc conservatively for the purposes of our framework:

      - Add the type alias `real` (same as `float`).
      - Add the predicates `int_ge`, `int_gt` mainly to simplify testing in lala_core.
      - Add the functions `int_minus`, `float_minus`, `int_neg`, `float_neg`.
      - Add the ability to have `true` and `false` in the `constraint` statement.
      - Parameters of predicates are not required to be flat, every constraint comes with a functional flavor, e.g., `int_le(int_plus(a,b), 5)` stands for `a+b <= 5`.
      - Several solve items are allowed, which is useful for multi-objectives optimization.
  */
  template<class Allocator>
  battery::shared_ptr<TFormula<Allocator>, Allocator> parse_flatzinc_str(const std::string& input, FlatZincOutput<Allocator>& output) {
    impl::FlatZincParser<Allocator> parser(output);
    return parser.parse(input);
  }

  template<class Allocator>
  battery::shared_ptr<TFormula<Allocator>, Allocator> parse_flatzinc(const std::string& filename, FlatZincOutput<Allocator>& output) {
    std::ifstream t(filename);
    std::string input((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    return parse_flatzinc_str<Allocator>(input, output);
  }

  template<class Allocator>
  battery::shared_ptr<TFormula<Allocator>, Allocator> parse_flatzinc_str(const std::string& input) {
    FlatZincOutput<Allocator> output;
    return parse_flatzinc_str(input, output);
  }

  template<class Allocator>
  battery::shared_ptr<TFormula<Allocator>, Allocator> parse_flatzinc(const std::string& filename) {
    FlatZincOutput<Allocator> output;
    return parse_flatzinc(filename, output);
  }
}

#endif
