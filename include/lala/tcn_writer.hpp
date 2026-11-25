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
    case ZADD: return "+";
    case ZMUL: return "*";
    case ZTDIV: case ZFDIV: case ZCDIV: case ZEDIV:
    case ZTMOD: case ZFMOD: case ZCMOD: case ZEMOD: return string_of_sig(op);
    case ZMAX: return "A";
    case ZMIN: return "I";
    case ZLEQ: return "L";
    case ZEQ: return "=";
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
 *    [ "minimize <obj>\n" ]                          # optional objective line
 *    <num_vars>\n
 *    <lb> <ub>\n                                     # one line per variable
 *    ...
 *    <num_constraints>\n
 *    <x> <y> <op> <z>\n                              # one line per ternary bytecode
 *    ...
 *
 *  `obj` is the index of the variable to minimize, or -1 for a satisfaction problem.
 *  Store and IProp are duck-typed: only the methods used below
 *  (vars(), num_deductions(), load_deduce(i)) need to be available.
 */
template <class Store, class IProp>
void write_preprocessed_tcn(std::ostream& out, const Store& store, const IProp& iprop, int obj = -1) {
  if(obj >= 0) {
    out << "minimize " << obj << '\n';
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

/** Append a search strategy to a TCN file: `search <var_order> <val_order> <k> <v_1> ... <v_k>`.
 *  Must be called after `write_preprocessed_tcn`, once per strategy, in the order they are applied.
 */
template <class Vars>
void write_tcn_search(std::ostream& out, const char* var_order, const char* val_order, const Vars& vars) {
  out << "search " << var_order << ' ' << val_order << ' ' << vars.size();
  for(int i = 0; i < vars.size(); ++i) {
    out << ' ' << vars[i].vid();
  }
  out << '\n';
}

}

#endif
