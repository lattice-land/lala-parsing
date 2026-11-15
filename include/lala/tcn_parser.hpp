// Copyright 2026 Wei Huang

#ifndef LALA_PARSING_TCN_PARSER_HPP
#define LALA_PARSING_TCN_PARSER_HPP

#include <fstream>
#include <string>
#include <iostream>
#include "battery/shared_ptr.hpp"
#include "lala/logic/ast.hpp"
#include "solver_output.hpp"

namespace lala {

namespace impl {

/** Map a TCN operator symbol to its Sig.
 *  Divisions and moduli use their lala names (tdiv, fdiv, cdiv, ediv, tmod, fmod, cmod, emod).
 */
inline Sig tcn_symbol_to_sig(const std::string& sym) {
  if(sym == "+") return ADD;
  if(sym == "*") return MUL;
  for(Sig sig : {TDIV, FDIV, CDIV, EDIV, TMOD, FMOD, CMOD, EMOD}) {
    if(sym == string_of_sig(sig)) return sig;
  }
  if(sym == "A") return MAX;
  if(sym == "I") return MIN;
  if(sym == "L") return LEQ;
  if(sym == "=") return EQ;
  std::cerr << "TCN parser: unknown operator '" << sym << "'" << std::endl;
  exit(EXIT_FAILURE);
}

} // namespace impl

/** Parse a Ternary Constraint Network (.tcn) file into a TFormula.
 *
 *  Format (see doc/tcn_parsing.md):
 *    [minimize K]   <- optional objective line (K is a variable index)
 *    N
 *    lb_0 ub_0
 *    ...
 *    lb_{N-1} ub_{N-1}
 *    M
 *    x_0 y_0 op_0 z_0
 *    ...
 *
 *  Variables are identified by their index in the file, and named x0, x1, ..., x{N-1} in the formula.
 *  Bounds: finite integer, "-inf" (no lower bound), or "inf" (no upper bound).
 *  Each constraint encodes: x_x = x_y op x_z.
 */
template <class Allocator>
battery::shared_ptr<TFormula<Allocator>, Allocator>
parse_tcn(const std::string& filename, SolverOutput<Allocator>& output)
{
  using F = TFormula<Allocator>;
  using FSeq = typename F::Sequence;

  std::ifstream in(filename);
  if(!in) {
    std::cerr << "TCN parser: cannot open file '" << filename << "'" << std::endl;
    return nullptr;
  }

  auto var_name = [](int i) -> std::string {
    return std::string("x") + std::to_string(i);
  };

  FSeq conjuncts{Allocator{}};

  // Peek at the first token to detect an optional objective line.
  std::string first;
  if(!(in >> first)) {
    std::cerr << "TCN parser: empty file" << std::endl;
    return nullptr;
  }

  const bool has_objective = first == "minimize";
  int obj_var = 0;
  // The objective is followed by the variable count.
  if(has_objective && !(in >> obj_var >> first)) {
    std::cerr << "TCN parser: expected a variable index after the objective" << std::endl;
    return nullptr;
  }

  int n = 0;
  try { n = std::stoi(first); }
  catch(...) {
    std::cerr << "TCN parser: expected number of variables, got '" << first << "'" << std::endl;
    return nullptr;
  }
  if(n < 0) {
    std::cerr << "TCN parser: negative number of variables" << std::endl;
    return nullptr;
  }
  if(has_objective && (obj_var < 0 || obj_var >= n)) {
    std::cerr << "TCN parser: objective variable index out of range" << std::endl;
    return nullptr;
  }

  for(int i = 0; i < n; ++i) {
    std::string lb_str, ub_str;
    if(!(in >> lb_str >> ub_str)) {
      std::cerr << "TCN parser: failed to read domain for variable x" << i << std::endl;
      return nullptr;
    }

    std::string name = var_name(i);

    // Declare variable as integer
    conjuncts.push_back(F::make_exists(UNTYPED,
      LVar<Allocator>(name.data()),
      Sort<Allocator>(Sort<Allocator>::Int)));

    // Lower bound constraint (skip when "-inf")
    if(lb_str != "-inf") {
      logic_int lb;
      try { lb = std::stoll(lb_str); }
      catch(...) {
        std::cerr << "TCN parser: invalid lower bound '" << lb_str
                  << "' for variable x" << i << std::endl;
        return nullptr;
      }
      conjuncts.push_back(F::make_binary(
        F::make_lvar(UNTYPED, LVar<Allocator>(name.data())),
        GEQ,
        F::make_z(lb)));
    }

    // Upper bound constraint (skip when "inf")
    if(ub_str != "inf") {
      logic_int ub;
      try { ub = std::stoll(ub_str); }
      catch(...) {
        std::cerr << "TCN parser: invalid upper bound '" << ub_str
                  << "' for variable x" << i << std::endl;
        return nullptr;
      }
      conjuncts.push_back(F::make_binary(
        F::make_lvar(UNTYPED, LVar<Allocator>(name.data())),
        LEQ,
        F::make_z(ub)));
    }
  }

  int m = 0;
  if(!(in >> m) || m < 0) {
    std::cerr << "TCN parser: failed to read number of constraints" << std::endl;
    return nullptr;
  }

  for(int i = 0; i < m; ++i) {
    int x, y, z;
    std::string op_str;
    if(!(in >> x >> y >> op_str >> z)) {
      std::cerr << "TCN parser: failed to read constraint " << i << std::endl;
      return nullptr;
    }
    if(x < 0 || x >= n || y < 0 || y >= n || z < 0 || z >= n) {
      std::cerr << "TCN parser: variable index out of range in constraint " << i << std::endl;
      return nullptr;
    }

    Sig op = impl::tcn_symbol_to_sig(op_str);
    // Encode: x_x = x_y op x_z
    conjuncts.push_back(F::make_binary(
      F::make_lvar(UNTYPED, LVar<Allocator>(var_name(x).data())),
      EQ,
      F::make_binary(
        F::make_lvar(UNTYPED, LVar<Allocator>(var_name(y).data())),
        op,
        F::make_lvar(UNTYPED, LVar<Allocator>(var_name(z).data())))));
  }

  // Objective goes last so all variables are declared before BAB interprets it.
  if(has_objective) {
    conjuncts.push_back(F::make_unary(MINIMIZE,
      F::make_lvar(UNTYPED, LVar<Allocator>(var_name(obj_var).data()))));
  }

  F formula = F::make_nary(AND, std::move(conjuncts));
  return battery::make_shared<F, Allocator>(std::move(formula));
}

} // namespace lala

#endif
