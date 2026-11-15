// Copyright 2026 Wei Huang

#ifndef LALA_PARSING_TCN_WRITER_HPP
#define LALA_PARSING_TCN_WRITER_HPP

#include <iostream>
#include <string>
#include "lala/logic/ast.hpp"

namespace lala {

/** Map a Sig to its TCN operator symbol.
 *  Divisions and moduli use their lala names (tdiv, fdiv, cdiv, ediv, tmod, fmod, cmod, emod).
 */
inline const char* tcn_sig_to_symbol(Sig op) {
  switch(op) {
    case ADD: return "+";
    case MUL: return "*";
    case TDIV: case FDIV: case CDIV: case EDIV:
    case TMOD: case FMOD: case CMOD: case EMOD: return string_of_sig(op);
    case MAX: return "A";
    case MIN: return "I";
    case LEQ: return "L";
    case EQ:  return "=";
    default:
      std::cerr << "Unsupported operator in TCN dump: " << string_of_sig_txt(op) << std::endl;
      exit(EXIT_FAILURE);
  }
}

/** Format the lower bound of a domain in the TCN text format.
 *  An open lower bound (top of the bound lattice) is rendered as "-inf";
 *  an empty lower bound (bot) is rendered as "inf".
 */
template <class Bound>
inline std::string tcn_lower_bound_to_string(const Bound& bound) {
  if(bound.is_top()) {
    return "-inf";
  }
  if(bound.is_bot()) {
    return "inf";
  }
  return std::to_string(bound.load());
}

/** Format the upper bound of a domain in the TCN text format.
 *  An open upper bound (top) is rendered as "inf"; an empty upper bound
 *  (bot) is rendered as "-inf".
 */
template <class Bound>
inline std::string tcn_upper_bound_to_string(const Bound& bound) {
  if(bound.is_top()) {
    return "inf";
  }
  if(bound.is_bot()) {
    return "-inf";
  }
  return std::to_string(bound.load());
}

/** Serialize an abstract state (variable domains + ternary propagators)
 *  to the TCN text format.
 *
 *  Layout:
 *    [ "minimize <vid>\n" | "maximize <vid>\n" ]     # optional objective line
 *    <num_vars>\n
 *    <lb> <ub>\n                                     # one line per variable
 *    ...
 *    <num_constraints>\n
 *    <x> <y> <op> <z>\n                              # one line per ternary bytecode
 *    ...
 *
 *  Store, IProp and BAB are duck-typed: only the methods used below
 *  (vars(), num_deductions(), load_deduce(i), is_minimization(),
 *  is_maximization(), objective_var()) need to be available.
 */
template <class Store, class IProp, class BAB>
void write_preprocessed_tcn(std::ostream& out, const Store& store, const IProp& iprop, const BAB& bab) {
  if(bab.is_minimization()) {
    out << "minimize " << bab.objective_var().vid() << '\n';
  }
  else if(bab.is_maximization()) {
    out << "maximize " << bab.objective_var().vid() << '\n';
  }
  out << store.vars() << '\n';
  for(int i = 0; i < store.vars(); ++i) {
    const auto& dom = store[i];
    out << tcn_lower_bound_to_string(dom.lb()) << ' '
        << tcn_upper_bound_to_string(dom.ub()) << '\n';
  }
  out << iprop.num_deductions() << '\n';
  for(int i = 0; i < iprop.num_deductions(); ++i) {
    auto bytecode = iprop.load_deduce(i);
    out << bytecode.x << ' '
        << bytecode.y << ' '
        << tcn_sig_to_symbol(bytecode.op) << ' '
        << bytecode.z << '\n';
  }
}

}

#endif
