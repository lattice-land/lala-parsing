// Copyright 2026 Pierre Talbot

#ifndef LALA_PARSING_TCN_WRITER_HPP
#define LALA_PARSING_TCN_WRITER_HPP

#include <iostream>
#include <string>
#include "lala/logic/ast.hpp"

namespace lala {

/** Map a Sig used by the AST to its canonical TCN operator symbol.
 *  Division variants (TDIV/FDIV/CDIV/EDIV) all serialize to "D";
 *  modulo variants (TMOD/FMOD/CMOD/EMOD/MOD) all serialize to "M".
 */
inline const char* tcn_sig_to_symbol(Sig op) {
  switch(op) {
    case ADD: return "+";
    case MUL: return "*";
    case TDIV:
    case FDIV:
    case CDIV:
    case EDIV: return "D";
    case MOD:
    case TMOD:
    case FMOD:
    case CMOD:
    case EMOD: return "M";
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
 *  an empty lower bound (bot) is rendered as "+inf".
 */
template <class Bound>
inline std::string tcn_lower_bound_to_string(const Bound& bound) {
  if(bound.is_top()) {
    return "-inf";
  }
  if(bound.is_bot()) {
    return "+inf";
  }
  return std::to_string(bound.value());
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
  return std::to_string(bound.value());
}

}

#endif
