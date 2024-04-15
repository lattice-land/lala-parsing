// Copyright 2022 Pierre Talbot

#ifndef LALA_PARSING_FLATZINC_PARSER_HPP
#define LALA_PARSING_FLATZINC_PARSER_HPP

#include "peglib.h"
#include <cassert>
#include <cstdlib>
#include <string>
#include <istream>
#include <fstream>
#include <streambuf>
#include <iostream>
#include <cfenv>
#include <set>
#include <cinttypes>

#include "battery/shared_ptr.hpp"
#include "lala/logic/ast.hpp"

namespace lala {

template<class Allocator>
class FlatZincOutput {
  using bstring = battery::string<Allocator>;
  template<class T> using bvector = battery::vector<T, Allocator>;
  using array_dim_t = bvector<battery::tuple<size_t,size_t>>;
  using F = TFormula<Allocator>;

  bvector<bstring> output_vars;
  // For each array, we store its output dimension characteristics and the list of the variables in the array.
  bvector<battery::tuple<bstring, array_dim_t, bvector<bstring>>> output_arrays;

public:
  template <class Alloc2>
  friend class FlatZincOutput;

  CUDA FlatZincOutput(const Allocator& alloc)
    : output_vars(alloc)
    , output_arrays(alloc)
  {}

  FlatZincOutput(FlatZincOutput&&) = default;
  FlatZincOutput<Allocator>& operator=(const FlatZincOutput<Allocator>&) = default;

  template <class Alloc>
  CUDA FlatZincOutput<Allocator>& operator=(const FlatZincOutput<Alloc>& other) {
    output_vars = other.output_vars;
    output_arrays = other.output_arrays;
    return *this;
  }

  template<class Alloc2>
  CUDA FlatZincOutput(const FlatZincOutput<Alloc2>& other, const Allocator& allocator = Allocator{})
    : output_vars(other.output_vars, allocator)
    , output_arrays(other.output_arrays, allocator)
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
      output_arrays.push_back(battery::make_tuple<bstring, array_dim_t, bvector<bstring>>(bstring(array_name), {}, {}));
      idx = static_cast<int>(output_arrays.size()) - 1;
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

  class SimplifierIdentity {
    template <class Alloc, class B, class Env>
    CUDA void print_variable(const LVar<Alloc>& vname, const Env& benv, const B& b) const {
      const auto& x = *(benv.variable_of(vname));
      x.sort.print_value(b.project(x.avars[0]));
    }
  };

  template <class Env, class A, class S>
  CUDA void print_solution(const Env& env, const A& sol, const S& simplifier = SimplifierIdentity{}) const {
    for(int i = 0; i < output_vars.size(); ++i) {
      printf("%s=", output_vars[i].data());
      simplifier.print_variable(output_vars[i], env, sol);
      printf(";\n");
    }
    for(int i = 0; i < output_arrays.size(); ++i) {
      const auto& dims = battery::get<1>(output_arrays[i]);
      const auto& array_vars = battery::get<2>(output_arrays[i]);
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
};

namespace impl {

/** Unfortunately, I'm really not sure this function works in all cases due to compiler bugs with rounding modes... */
inline logic_real string_to_real(const std::string& s) {
  #if !defined(__GNUC__) && !defined(_MSC_VER)
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
  using Set = logic_set<F>;
  using So = Sort<allocator_type>;
  using bstring = battery::string<Allocator>;
  using FSeq = typename F::Sequence;

  std::map<std::string, F> params; // Name and value of the parameters occuring in the model.
  std::map<std::string, int> arrays; // Size of all named arrays (parameters and variables).
  bool error; // If an error was found during parsing.
  bool silent; // If we do not want to output error messages.
  FlatZincOutput<Allocator>& output;

  // Contains all the annotations ignored.
  // It is used to avoid printing an error message more than once per annotation.
  std::set<std::string> ignored_annotations;

  enum class TableKind {
    PLAIN,
    SHORT,
    COMPRESSED,
    BASIC
  };

public:
  FlatZincParser(FlatZincOutput<Allocator>& output): error(false), silent(false), output(output) {}

  battery::shared_ptr<F, allocator_type> parse(const std::string& input) {
    peg::parser parser(R"(
      Statements  <- (PredicateDecl / VariableDecl / VarArrayDecl / ParameterDecl / ConstraintDecl / SolveItem / Comment)+

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

      PredicateDecl <- 'predicate' Identifier '(' (!')' .)* ')' ';'

      MinimizeItem <- 'minimize' RangeLiteral
      MaximizeItem <- 'maximize' RangeLiteral
      SatisfyItem <- 'satisfy'
      SolveItem <- 'solve' SearchAnnotations (MinimizeItem / MaximizeItem / SatisfyItem) ';'

      SearchAnnotations <- ('::' SearchAnnotation)*
      SearchAnnotation <- SeqSearch / BaseSearch
      SeqSearch <- 'seq_search' '(' '[' SearchAnnotation (',' SearchAnnotation)* ']' ')'
      BaseSearch <- ('int_search' / 'bool_search' / 'set_search') '(' (VariableLit / LiteralArray) ',' Identifier ',' Identifier ',' Identifier ')'

      LiteralArray <- '[]' / '[' RangeLiteral (',' RangeLiteral)* ']'
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
    parser["PredicateDecl"] = [this] (const SV &sv) { return F(); };
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
    parser["SolveItem"] = [this](const SV &sv) { return make_solve_item(sv);};
    parser["SearchAnnotations"] = [this](const SV &sv) { return make_search_annotations(sv);};
    parser["BaseSearch"] = [this](const SV &sv) { return make_base_search(sv);};
    parser["SeqSearch"] = [this](const SV &sv) { return make_seq_search(sv);};
    parser.set_logger([](size_t line, size_t col, const std::string& msg, const std::string &rule) {
      std::cerr << line << ":" << col << ": " << msg << "\n";
    });

    F f;
    if(parser.parse(input.c_str(), f) && !error) {
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
        "` expects `" + std::to_string(expected) + "` parameters" +
        ", but we got `" + std::to_string(obtained) + "` parameters.");
    }

    F make_set_literal(const SV& sv) {
      logic_set<F> set;
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
              set.push_back(battery::make_tuple(params[name], params[name]));
              element_added = true;
            }
            else {
              return make_error(sv, "Undeclared parameter `" + name + "`.");
            }
          }
          if(!element_added) {
            set.push_back(battery::make_tuple(element, element));
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
        idx = static_cast<int>(index.z());
      }
      else if(index.is(F::LV)) {
        if(params.contains(index.lv().data())) {
          auto pindex = params[index.lv().data()];
          if(pindex.is(F::Z)) {
            idx = static_cast<int>(pindex.z());
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
        arrays[identifier] = static_cast<int>(array.size());
        for(int i = 0; i < static_cast<int>(array.size()); ++i) {
          auto id = make_array_access(identifier, i);
          params[id] = f(array[i]);
        }
      }
      return F::make_true();
    }

    F make_variable_array_decl(const SV& sv) {
      int arraySize = static_cast<int>(f(sv[0]).z());
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
          AType ty = f(annot[1]).z(); // assignment of logic_int (int64_t) to ty (int) truncates ty (bug?)
          formula.type_as(ty);
        }
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
          if(!ignored_annotations.contains(name)) {
            ignored_annotations.insert(name);
            std::cerr << "% WARNING: Annotation " + name + " is unknown and was ignored." << std::endl;
          }
        }
      }
      return std::move(formula);
    }

    F make_binary(Sig sig, const SV &sv) {
      if(sv.size() != 3) {
        return make_arity_error(sv, sig, 2, static_cast<int>(sv.size()/*size_t*/) - 1);
      }
      return F::make_binary(f(sv[1]), sig, f(sv[2]));
    }

    F make_unary_fun_eq(Sig sig, const SV &sv, Sig eq_kind = EQ) {
      if(sv.size() != 3) {
        return make_arity_error(sv, sig, 1, static_cast<int>(sv.size()) - 2);
      }
      auto fun = F::make_unary(sig, f(sv[1]));
      return F::make_binary(fun, eq_kind, f(sv[2]));
    }

    F make_unary_fun(Sig sig, const SV &sv) {
      if(sv.size() != 2) {
        return make_arity_error(sv, sig, 1, static_cast<int>(sv.size()) - 1);
      }
      return F::make_unary(sig, f(sv[1]));
    }

    F make_binary_fun_eq(Sig sig, const SV &sv, Sig eq_kind = EQ) {
      if(sv.size() != 4) {
        return make_arity_error(sv, sig, 2, static_cast<int>(sv.size()) - 2);
      }
      auto left = F::make_binary(f(sv[1]), sig, f(sv[2]));
      auto right = f(sv[3]);
      if(eq_kind == EQUIV && right.is_true()) {
        return left;
      }
      return F::make_binary(left, eq_kind, right);
    }

    F make_binary_fun(Sig sig, const SV& sv) {
      if(sv.size() != 3) {
        return make_arity_error(sv, sig, 2, static_cast<int>(sv.size()) - 1);
      }
      return F::make_binary(f(sv[1]), sig, f(sv[2]));
    }

    F make_nary_fun(Sig sig, const SV &sv) {
      FSeq seq;
      for(int i = 1; i < sv.size(); ++i) {
        seq.push_back(f(sv[i]));
      }
      return F::make_nary(sig, std::move(seq));
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
      if (name == "int_le") { return make_binary(LEQ, sv); }
      if (name == "int_lt") { return make_binary(LT, sv); }
      if (name == "int_ge") { return make_binary(GEQ, sv); }
      if (name == "int_gt") { return make_binary(GT, sv); }
      if (name == "int_eq") { return make_binary(EQ, sv); }
      if (name == "int_ne") { return make_binary(NEQ, sv); }
      if (name == "int_abs") { return make_unary_fun_eq(ABS, sv); }
      if (name == "int_neg") { return make_unary_fun_eq(NEG, sv); }
      if (name == "int_div") { return make_binary_fun_eq(EDIV, sv); }
      if (name == "int_mod") { return make_binary_fun_eq(EMOD, sv); }
      if (name == "int_plus") { return make_binary_fun_eq(ADD, sv); }
      if (name == "int_minus") { return make_binary_fun_eq(SUB, sv); }
      if (name == "int_pow") { return make_binary_fun_eq(POW, sv); }
      if (name == "int_times") { return make_binary_fun_eq(MUL, sv); }
      if (name == "int_max") { return make_binary_fun_eq(MAX, sv); }
      if (name == "int_min") { return make_binary_fun_eq(MIN, sv); }
      if (name == "int_eq_reif") { return make_binary_fun_eq(EQ, sv, EQUIV); }
      if (name == "int_le_reif") { return make_binary_fun_eq(LEQ, sv, EQUIV); }
      if (name == "int_lt_reif") { return make_binary_fun_eq(LT, sv, EQUIV); }
      if (name == "int_ne_reif") { return make_binary_fun_eq(NEQ, sv, EQUIV); }
      if (name == "bool2int") { return make_binary(EQ, sv); }
      if (name == "bool_eq") { return make_binary(EQ, sv); }
      if (name == "bool_le") { return make_binary(LEQ, sv); }
      if (name == "bool_lt") { return make_binary(LT, sv); }
      if (name == "bool_eq_reif") { return make_binary_fun_eq(EQ, sv, EQUIV); }
      if (name == "bool_le_reif") { return make_binary_fun_eq(LEQ, sv, EQUIV); }
      if (name == "bool_lt_reif") { return make_binary_fun_eq(LT, sv, EQUIV); }
      if (name == "bool_and") { return make_binary_fun_eq(AND, sv, EQUIV); }
      if (name == "bool_not") {
        if(sv.size() == 3) { return make_binary(XOR, sv); }
        else { return make_unary_fun(NOT, sv); }
      }
      if (name == "bool_or") { return make_binary_fun_eq(OR, sv, EQUIV); }
      if (name == "nbool_and") { return make_nary_fun(AND, sv); }
      if (name == "nbool_or") { return make_nary_fun(OR, sv); }
      if (name == "nbool_equiv") { return make_nary_fun(EQUIV, sv); }
      if (name == "bool_xor") {
        if(sv.size() == 3) { return make_binary(XOR, sv); }
        else { return make_binary_fun_eq(XOR, sv, EQUIV); }
      }
      if (name == "set_card") { return make_unary_fun_eq(CARD, sv); }
      if (name == "set_diff") { return make_binary_fun_eq(DIFFERENCE, sv); }
      if (name == "set_eq") { return make_binary(EQ, sv); }
      if (name == "set_eq_reif") { return make_binary_fun_eq(EQ, sv, EQUIV); }
      if (name == "set_in") { return make_binary(IN, sv); }
      if (name == "set_in_reif") { return make_binary_fun_eq(IN, sv, EQUIV); }
      if (name == "set_intersect") { return make_binary_fun_eq(INTERSECTION, sv, EQUIV); }
      if (name == "set_union") { return make_binary_fun_eq(UNION, sv, EQUIV); }
      if (name == "set_ne") { return make_binary(NEQ, sv); }
      if (name == "set_ne_reif") { return make_binary_fun_eq(NEQ, sv, EQUIV); }
      if (name == "set_subset") { return make_binary(SUBSETEQ, sv); }
      if (name == "set_subset_reif") { return make_binary_fun_eq(SUBSETEQ, sv, EQUIV); }
      if (name == "set_superset") { return make_binary(SUPSETEQ, sv); }
      if (name == "set_symdiff") { return make_binary_fun_eq(SYMMETRIC_DIFFERENCE, sv, EQUIV); }
      if (name == "set_le") { return make_binary(LEQ, sv); }
      if (name == "set_le_reif") { return make_binary_fun_eq(LEQ, sv, EQUIV); }
      if (name == "set_lt") { return make_binary(LT, sv); }
      if (name == "set_lt_reif") { return make_binary_fun_eq(LT, sv, EQUIV); }
      if (name == "float_abs") { return make_binary_fun_eq(ABS, sv); }
      if (name == "float_neg") { return make_binary_fun_eq(NEG, sv); }
      if (name == "float_plus") { return make_binary_fun_eq(ADD, sv); }
      if (name == "float_minus") { return make_binary_fun_eq(SUB, sv); }
      if (name == "float_times") { return make_binary_fun_eq(MUL, sv); }
      if (name == "float_acos") { return make_unary_fun_eq(ACOS, sv); }
      if (name == "float_acosh") { return make_unary_fun_eq(ACOSH, sv); }
      if (name == "float_asin") { return make_unary_fun_eq(ASIN, sv); }
      if (name == "float_asinh") { return make_unary_fun_eq(ASINH, sv); }
      if (name == "float_atan") { return make_unary_fun_eq(ATAN, sv); }
      if (name == "float_atanh") { return make_unary_fun_eq(ATANH, sv); }
      if (name == "float_cos") { return make_unary_fun_eq(COS, sv); }
      if (name == "float_cosh") { return make_unary_fun_eq(COSH, sv); }
      if (name == "float_sin") { return make_unary_fun_eq(SIN, sv); }
      if (name == "float_sinh") { return make_unary_fun_eq(SINH, sv); }
      if (name == "float_tan") { return make_unary_fun_eq(TAN, sv); }
      if (name == "float_tanh") { return make_unary_fun_eq(TANH, sv); }
      if (name == "float_div") { return make_binary(DIV, sv); }
      if (name == "float_eq") { return make_binary(EQ, sv); }
      if (name == "float_eq_reif") { return make_binary_fun_eq(EQ, sv, EQUIV); }
      if (name == "float_le") { return make_binary(LEQ, sv); }
      if (name == "float_le_reif") { return make_binary_fun_eq(LEQ, sv, EQUIV); }
      if (name == "float_ne") { return make_binary(NEQ, sv); }
      if (name == "float_ne_reif") { return make_binary_fun_eq(NEQ, sv, EQUIV); }
      if (name == "float_lt") { return make_binary(LT, sv); }
      if (name == "float_lt_reif") { return make_binary_fun_eq(LT, sv, EQUIV); }
      if (name == "float_in") { return make_float_in(sv); }
      if (name == "float_in_reif") {
        return F::make_binary(make_float_in(sv), EQUIV, f(sv[4]));
      }
      if (name == "float_log10") { return make_log_eq(10, sv); }
      if (name == "float_log2") { return make_log_eq(2, sv); }
      if (name == "float_min") { return make_binary_fun_eq(MIN, sv); }
      if (name == "float_max") { return make_binary_fun_eq(MAX, sv); }
      if (name == "float_exp") { return make_unary_fun_eq(EXP, sv); }
      if (name == "float_ln") { return make_unary_fun_eq(LN, sv); }
      if (name == "float_pow") { return make_binary_fun_eq(POW, sv); }
      if (name == "float_sqrt") { return make_unary_fun_eq(SQRT, sv); }
      if (name == "int2float") { return make_binary(EQ, sv); }
      if (name == "array_int_element" || name == "array_var_int_element"
        || name == "array_bool_element" || name == "array_var_bool_element"
        || name == "array_set_element" || name == "array_var_set_element"
        || name == "array_float_element" || name == "array_var_float_element")
      {
        return make_element_constraint(name, sv);
      }
      if (name == "int_lin_eq" || name == "bool_lin_eq" || name == "float_lin_eq" ||
              name == "int_lin_eq_reif" || name == "bool_lin_eq_reif" || name == "float_lin_eq_reif")
      {
        return make_linear_constraint(name, EQ, sv);
      }
      if (name == "int_lin_le" || name == "bool_lin_le" || name == "float_lin_le" ||
              name == "int_lin_le_reif" || name == "bool_lin_le_reif" || name == "float_lin_le_reif")
      {
        return make_linear_constraint(name, LEQ, sv);
      }
      if (name == "int_lin_ne" || name == "bool_lin_ne" || name == "float_lin_ne" ||
              name == "int_lin_ne_reif" || name == "bool_lin_ne_reif" || name == "float_lin_ne_reif")
      {
        return make_linear_constraint(name, NEQ, sv);
      }
      if (name == "array_bool_and") { return make_boolean_constraint(name, AND, sv); }
      if (name == "array_bool_or") { return make_boolean_constraint(name, OR, sv); }
      if (name == "array_bool_xor") { return make_boolean_constraint(name, XOR, sv); }
      if (name == "bool_clause" || name == "bool_clause_reif") {
        return make_boolean_clause(name, sv);
      }
      if(name == "turbo_fzn_table_bool" || name == "turbo_fzn_table_int") {
        return make_table_constraint(name, sv, TableKind::PLAIN);
      }
      if(name == "turbo_fzn_short_table_int" || name == "turbo_fzn_short_table_set_of_int") {
        return make_table_constraint(name, sv, TableKind::SHORT);
      }
      if(name == "turbo_fzn_basic_table_int" || name == "turbo_fzn_basic_table_set_of_int") {
        return make_table_constraint(name, sv, TableKind::BASIC);
      }
      if(name == "turbo_fzn_compressed_table_int") {
        return make_table_constraint(name, sv, TableKind::COMPRESSED);
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
        if (name == "int_neg") { return make_unary_fun(NEG, sv); }
        if (name == "int_div") { return make_binary_fun(EDIV, sv); }
        if (name == "int_mod") { return make_binary_fun(EMOD, sv); }
        if (name == "int_plus") { return make_binary_fun(ADD, sv); }
        if (name == "int_minus") { return make_binary_fun(SUB, sv); }
        if (name == "int_pow") { return make_binary_fun(POW, sv); }
        if (name == "int_times") { return make_binary_fun(MUL, sv); }
        if (name == "int_max") { return make_binary_fun(MAX, sv); }
        if (name == "int_min") { return make_binary_fun(MIN, sv); }
        if (name == "bool_and") { return make_binary_fun(AND, sv); }
        if (name == "bool_not") { return make_unary_fun(NOT, sv); }
        if (name == "bool_or") { return make_binary_fun(OR, sv); }
        if (name == "bool_xor") { return make_binary_fun(XOR, sv); }
        if (name == "set_card") { return make_unary_fun(CARD, sv); }
        if (name == "set_diff") { return make_binary_fun(DIFFERENCE, sv); }
        if (name == "set_intersect") { return make_binary_fun(INTERSECTION, sv); }
        if (name == "set_union") { return make_binary_fun(UNION, sv); }
        if (name == "set_symdiff") { return make_binary_fun(SYMMETRIC_DIFFERENCE, sv); }
        if (name == "float_abs") { return make_binary_fun(ABS, sv); }
        if (name == "float_neg") { return make_binary_fun(NEG, sv); }
        if (name == "float_plus") { return make_binary_fun(ADD, sv); }
        if (name == "float_minus") { return make_binary_fun(SUB, sv); }
        if (name == "float_times") { return make_binary_fun(MUL, sv); }
        if (name == "float_acos") { return make_unary_fun(ACOS, sv); }
        if (name == "float_acosh") { return make_unary_fun(ACOSH, sv); }
        if (name == "float_asin") { return make_unary_fun(ASIN, sv); }
        if (name == "float_asinh") { return make_unary_fun(ASINH, sv); }
        if (name == "float_atan") { return make_unary_fun(ATAN, sv); }
        if (name == "float_atanh") { return make_unary_fun(ATANH, sv); }
        if (name == "float_cos") { return make_unary_fun(COS, sv); }
        if (name == "float_cosh") { return make_unary_fun(COSH, sv); }
        if (name == "float_sin") { return make_unary_fun(SIN, sv); }
        if (name == "float_sinh") { return make_unary_fun(SINH, sv); }
        if (name == "float_tan") { return make_unary_fun(TAN, sv); }
        if (name == "float_tanh") { return make_unary_fun(TANH, sv); }
        if (name == "float_div") { return make_binary_fun(DIV, sv); }
        if (name == "float_log10") { return make_log(10, sv); }
        if (name == "float_log2") { return make_log(2, sv); }
        if (name == "float_min") { return make_binary_fun(MIN, sv); }
        if (name == "float_max") { return make_binary_fun(MAX, sv); }
        if (name == "float_exp") { return make_unary_fun(EXP, sv); }
        if (name == "float_ln") { return make_unary_fun(LN, sv); }
        if (name == "float_pow") { return make_binary_fun(POW, sv); }
        if (name == "float_sqrt") { return make_unary_fun(SQRT, sv); }
        return make_error(sv, "Unknown function or predicate symbol `" + name + "`");
      }
    }

    F make_statements(const SV& sv) {
      if(sv.size() == 1) {
        return f(sv[0]);
      }
      else {
        FSeq children;
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
        ty);
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
      FSeq seq;
      for(int i = 0; i < array.seq().size(); ++i) {
        // index = (i+1) ==> varName = value
        seq.push_back(F::make_binary(
          F::make_binary(index, EQ, F::make_z(i+1)),
          IMPLY,
          F::make_binary(array.seq(i), EQ, value)));
      }
      return F::make_nary(AND, std::move(seq));
    }

    F resolve_bool(const SV& sv, const std::any& any) {
      try {
        auto boolParam = f(any);
        if(boolParam.is(F::LV)) {
          std::string paramName(boolParam.lv().data());
          if(params.contains(paramName)) {
            boolParam = params[paramName];
          }
          else {
            return make_error(sv, "Undeclared parameter `" + paramName + "`.");
          }
        }
        if(boolParam.is(F::B)) {
          return boolParam;
        }
        else {
          return make_error(sv, "Expects a Boolean parameter.");
        }
      }
      catch(std::bad_any_cast) {
        return make_error(sv, "Expects a Boolean parameter.");
      }
    }

    // We return the elements inside the array in the form `arr[1] /\ arr[2] /\ ... /\ arr[N]`.
    // The array can either be a literal array directly, or the name of an array.
    F resolve_array(const SV& sv, const std::any& any) {
      FSeq seq;
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
        try {
          auto array = std::any_cast<SV>(any);
          for(int i = 0; i < array.size(); ++i) {
            seq.push_back(f(array[i]));
          }
        }
        catch(std::bad_any_cast) {
          return make_error(sv, "Expects an array of valid elements.");
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
      FSeq sum;
      for(int i = 0; i < as.seq().size(); ++i) {
        sum.push_back(F::make_binary(as.seq(i), MUL, bs.seq(i)));
      }
      F linearCons =
          sum.size() == 1
        ? F::make_binary(std::move(sum[0]), sig, c)
        : F::make_binary(F::make_nary(ADD, std::move(sum)), sig, c);
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
      FSeq negs;
      for(int i = 0; i < bs.seq().size(); ++i) {
        negs.push_back(F::make_unary(NOT, bs.seq(i)));
      }
      F clause = F::make_binary(F::make_nary(OR, as.seq()), OR, F::make_nary(OR, negs));
      if(sv.size() == 4) {
        return F::make_binary(f(sv[3]), EQUIV, std::move(clause));
      }
      else {
        return std::move(clause);
      }
    }

    /** Given a cell `f` in the context of a table constraint, and `sort` the type of the variable.
     * If `f` represents the values taken by an integer variable, then 2147483647 and {} are the wildcards.
     * If `f` represents the values taken by a set of integer variable, then {2147483647} is the wildcard.
     */
    bool is_wildcard(const F& f, So sort) {
      if(sort.is_int()) {
        return (f.is(F::Z) && f.z() == 2147483647)
            || (f.is(F::S) && f.s().size() == 0);
      }
      else if(sort.is_set() && f.is(F::S) && f.s().size() == 1) {
        auto l = battery::get<0>(f.s()[0]);
        auto u = battery::get<1>(f.s()[0]);
        return l.is(F::Z) && u.is(F::Z) && l.z() == 2147483647 && u.z() == 2147483647;
      }
      return false;
    }

    So sort_of_table_constraint(const std::string& name) {
      if(name == "turbo_fzn_table_bool") { return So(So::Bool); }
      else if(name == "turbo_fzn_table_int") { return So(So::Int); }
      else if(name == "turbo_fzn_short_table_int") { return So(So::Int); }
      else if(name == "turbo_fzn_short_table_set_of_int") { return So(So::Set, So(So::Int)); }
      else if(name == "turbo_fzn_basic_table_int") { return So(So::Int); }
      else if(name == "turbo_fzn_basic_table_set_of_int") { return So(So::Set, So(So::Int)); }
      else if(name == "turbo_fzn_compressed_table_int") { return So(So::Int); }
      else { printf("missing table constraint."); assert(false); return So(So::Bool); }
    }

    F make_table_constraint(const std::string& name, const SV& sv, TableKind kind) {
      bool positive = true;
      int header_idx = 1;
      int table_idx = 2;
      if(sv.size() != 3 && (name == "turbo_fzn_table_bool" || name == "turbo_fzn_table_bool")) {
        return make_error(sv, "`" + name + "` expects 2 parameters, but we got `" + std::to_string(sv.size() - 1) + "` parameters");
      }
      else {
        if(sv.size() != 4) {
          return make_error(sv, "`" + name + "` expects 3 parameters, but we got `" + std::to_string(sv.size() - 1) + "` parameters");
        }
        auto boolParam = resolve_bool(sv, sv[1]);
        if(!boolParam.is(F::B)) { return boolParam; }
        positive = boolParam.b();
        ++header_idx;
        ++table_idx;
      }
      auto header = resolve_array(sv, sv[header_idx]);
      if(!header.is(F::Seq)) { return header; }
      auto table = resolve_array(sv, sv[table_idx]);
      if(!table.is(F::Seq)) { return table; }
      size_t num_t_cols = (kind == TableKind::BASIC) ? header.seq().size() * 2 : header.seq().size();
      if(table.seq().size() % num_t_cols != 0) {
        return make_error(sv, "`" + name + "` expects the number of variables is equal to the number of columns of the table (or twice for basic tables).");
      }
      size_t num_cols = header.seq().size();
      size_t num_rows = table.seq().size() / header.seq().size();
      So sort = sort_of_table_constraint(name);
      FSeq disjuncts;
      for(int i = 0; i < num_rows; ++i) {
        FSeq conjuncts;
        for(int z = 0; z < num_cols; ++z) {
          int j = z * (kind == TableKind::BASIC ? 2 : 1);
          auto cell = table.seq(i*num_t_cols + j);
          auto var = header.seq(z);
          switch(kind) {
            case TableKind::SHORT: {
              if(is_wildcard(cell, sort)) { continue; }
            }
            case TableKind::PLAIN: { // x[j] == t[i][j]
              if(!cell.is(F::S) && !cell.is(F::Z) && !cell.is(F::B)) {
                return make_error(sv, "`" + name + "` expects each cell to be an integer, a set or a Boolean.");
              }
              conjuncts.push_back(F::make_binary(var, EQ, cell));
              break;
            }
            case TableKind::COMPRESSED: {
              if(!cell.is(F::S)) {
                return make_error(sv, "`" + name + "` expects each cell to be a set.");
              }
              if(is_wildcard(cell, sort)) { continue; }
              conjuncts.push_back(F::make_binary(var, IN, cell));
              break;
            }
            case TableKind::BASIC: {
              auto l = cell;
              auto u = table.seq(i*num_t_cols + j + 1);
              if(!l.is(F::S) && !l.is(F::Z) && !u.is(F::S) && !u.is(F::Z)) {
                return make_error(sv, "`" + name + "` expects each cell to be an integer or a set.");
              }
              if(!is_wildcard(l, sort)) {
                if(l.is(F::Z)) {
                  conjuncts.push_back(F::make_binary(var, GEQ, l));
                }
                else {
                  conjuncts.push_back(F::make_binary(var, SUPSETEQ, l));
                }
              }
              if(!is_wildcard(u, sort)) {
                if(u.is(F::Z)) {
                  conjuncts.push_back(F::make_binary(var, LEQ, u));
                }
                else {
                  conjuncts.push_back(F::make_binary(var, SUBSETEQ, u));
                }
              }
              break;
            }
          }
        }
        if(conjuncts.size() == 0) {
          return F::make_true();
        }
        else if(conjuncts.size() == 1) {
          disjuncts.push_back(std::move(conjuncts[0]));
        }
        else {
          disjuncts.push_back(F::make_nary(AND, std::move(conjuncts)));
        }
      }
      if(disjuncts.size() == 0) {
        return F::make_false();
      }
      else if(disjuncts.size() == 1) {
        return std::move(disjuncts[0]);
      }
      else {
        return F::make_nary(OR, std::move(disjuncts));
      }
    }

    F make_solve_item(const SV& sv) {
      if(sv.size() == 1) {
        return f(sv[0]);
      }
      else {
        if(f(sv[1]).is_true()) {
          return f(sv[0]);
        }
        else {
          return F::make_binary(f(sv[0]), AND, f(sv[1]));
        }
      }
    }

    void make_seq_search(const SV& sv, FSeq& seq) {
      for(int i = 0; i < sv.size(); ++i) {
        const auto& sub_search = f(sv[i]);
        if(sub_search.is(F::Seq) && sub_search.sig() == AND) {
          for(int j = 0; j < sub_search.seq().size(); ++j) {
            seq.push_back(sub_search.seq(j));
          }
        }
        else {
          seq.push_back(sub_search);
        }
      }
    }

    F make_seq_search(const SV& sv) {
      FSeq seq;
      make_seq_search(sv, seq);
      return F::make_nary(AND, seq);
    }

    F make_base_search(const SV& sv) {
      FSeq seq;
      seq.push_back(F::make_nary(bstring(std::any_cast<std::string>(sv[1]).data()), FSeq{}));
      seq.push_back(F::make_nary(bstring(std::any_cast<std::string>(sv[2]).data()), FSeq{}));
      auto array = resolve_array(sv, sv[0]);
      for(int i = 0; i < array.seq().size(); ++i) {
        if(array.seq(i).is_variable()) {
          seq.push_back(array.seq(i));
        }
      }
      return F::make_nary("search", seq);
    }

    F make_search_annotations(const SV& sv) {
      if(sv.size() == 1) {
        return f(sv[0]);
      }
      else {
        return make_seq_search(sv);
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
    if(t.is_open()) {
      std::string input((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
      return parse_flatzinc_str<Allocator>(input, output);
    }
    else {
      std::cerr << "File `" << filename << "` does not exists:." << std::endl;
    }
    return nullptr;
  }

  template<class Allocator>
  battery::shared_ptr<TFormula<Allocator>, Allocator> parse_flatzinc_str(const std::string& input, const Allocator& allocator = Allocator()) {
    FlatZincOutput<Allocator> output(allocator);
    return parse_flatzinc_str(input, output);
  }

  template<class Allocator>
  battery::shared_ptr<TFormula<Allocator>, Allocator> parse_flatzinc(const std::string& filename, const Allocator& allocator = Allocator()) {
    FlatZincOutput<Allocator> output(allocator);
    return parse_flatzinc(filename, output);
  }
}

#endif
