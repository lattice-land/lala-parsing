// Copyright 2022 Pierre Talbot

#ifndef LALA_PARSING_XCSP3_PARSER_HPP
#define LALA_PARSING_XCSP3_PARSER_HPP

#include <iostream>
#include <optional>

#include "XCSP3Tree.h"
#include "XCSP3CoreCallbacks.h"
#include "XCSP3Variable.h"
#include "XCSP3CoreParser.h"

#include "lala/logic/ast.hpp"
#include "battery/shared_ptr.hpp"

#include "flatzinc_parser.hpp"
#include "solver_output.hpp"

namespace XCSP3Core {
  template <class Allocator>
  class XCSP3_turbo_callbacks;
}

namespace lala {
  enum class TableDecomposition {
    DISJUNCTIVE,
    TABLE_PREDICATE,
    ELEMENTS
  };


 template<class Allocator>
  void parse_xcsp3(const std::string& filename, XCSP3Core::XCSP3_turbo_callbacks<Allocator> &cb) {
    ::XCSP3Core::XCSP3CoreParser parser(&cb);
    parser.parse(filename.c_str());
  }

  template<class Allocator>
  battery::shared_ptr<TFormula<Allocator>, Allocator> parse_xcsp3(const std::string& filename, SolverOutput<Allocator>& output, TableDecomposition d = TableDecomposition::ELEMENTS) {
    ::XCSP3Core::XCSP3_turbo_callbacks<Allocator> cb(output, d);
    parse_xcsp3(filename, cb);
    return cb.build_formula();
  }

  template<class Allocator>
  battery::shared_ptr<TFormula<Allocator>, Allocator> parse_xcsp3_str(const std::string& input, TableDecomposition d = TableDecomposition::ELEMENTS, const Allocator& allocator = Allocator()) {
    SolverOutput<Allocator> output(allocator, OutputType::XCSP);
    return parse_xcsp3_str(input, output, d);
  }

  template<class Allocator>
  battery::shared_ptr<TFormula<Allocator>, Allocator> parse_xcsp3(const std::string& filename, TableDecomposition d = TableDecomposition::ELEMENTS, const Allocator& allocator = Allocator()) {
    SolverOutput<Allocator> output(allocator, OutputType::XCSP);
    return parse_xcsp3(filename, output, d);
  }
}

/**
 * This is an example that prints useful informations of a XCSP3 instance.
 * You need to create your own class and to override functions of the callback.
 * We suggest to make a map between XVariable and your own variables in order to
 * facilitate the constructions of constraints.
 *
 * see main.cc to show declaration of the parser
 *
 */

namespace XCSP3Core {
    using namespace std;

    template <class Allocator>
    class XCSP3_turbo_callbacks : public XCSP3CoreCallbacks {
      using allocator_type = Allocator;
      using F = lala::TFormula<allocator_type>;
      using FSeq = typename F::Sequence;

    public:
      using XCSP3CoreCallbacks::buildConstraintMinimum;
      using XCSP3CoreCallbacks::buildConstraintMaximum;
      using XCSP3CoreCallbacks::buildConstraintElement;
      using XCSP3CoreCallbacks::buildObjectiveMinimize;
      using XCSP3CoreCallbacks::buildObjectiveMaximize;

      XCSP3_turbo_callbacks(::lala::SolverOutput<Allocator>& output, lala::TableDecomposition d = lala::TableDecomposition::ELEMENTS):
        XCSP3CoreCallbacks(), canonize(true), output(output), table_decomposition(d) {}

      virtual void beginInstance(InstanceType type) override;
      virtual void endInstance() override;
      virtual void beginVariables() override;
      virtual void endVariables() override;
      virtual void beginVariableArray(string id) override;
      virtual void endVariableArray() override;
      virtual void beginConstraints() override;
      virtual void endConstraints() override;
      virtual void beginGroup(string id) override;
      virtual void endGroup() override;
      virtual void beginBlock(string classes) override;
      virtual void endBlock() override;
      virtual void beginSlide(string id, bool circular) override;
      virtual void endSlide() override;
      virtual void beginObjectives() override;
      virtual void endObjectives() override;
      virtual void beginAnnotations() override;
      virtual void endAnnotations() override;
      virtual void buildVariableInteger(string id, int minValue, int maxValue) override;
      virtual void buildVariableInteger(string id, vector<int> &values) override;
      virtual void buildConstraintExtension(string id, vector<XVariable *> list, vector<vector<int>> &tuples, bool support, bool hasStar) override;
      virtual void buildConstraintExtension(string id, XVariable *variable, vector<int> &tuples, bool support, bool hasStar) override;

      virtual void buildConstraintExtensionAs(string id, vector<XVariable *> list, bool support, bool hasStar) override;
      virtual void buildConstraintIntension(string id, string expr) override;
      virtual void buildConstraintIntension(string id, Tree *tree) override;
      virtual void buildConstraintPrimitive(string id, OrderType op, XVariable *x, int k, XVariable *y) override;
      virtual void buildConstraintPrimitive(string id, OrderType op, XVariable *x, int k) override;
      virtual void buildConstraintPrimitive(string id, XVariable *x,  bool in, int min, int max) override;
      virtual void buildConstraintRegular(string id, vector<XVariable *> &list, string st, vector<string> &final, vector<XTransition> &transitions) override;
      virtual void buildConstraintMDD(string id, vector<XVariable *> &list, vector<XTransition> &transitions) override;
      virtual void buildConstraintAlldifferent(string id, vector<XVariable *> &list) override;
      virtual void buildConstraintAlldifferentExcept(string id, vector<XVariable *> &list, vector<int> &except) override;
      virtual void buildConstraintAlldifferent(string id, vector<Tree *> &list) override;
      virtual void buildConstraintAlldifferentList(string id, vector<vector<XVariable *>> &lists) override;
      virtual void buildConstraintAlldifferentMatrix(string id, vector<vector<XVariable *>> &matrix) override;
      virtual void buildConstraintAllEqual(string id, vector<XVariable *> &list) override;
      virtual void buildConstraintNotAllEqual(string id, vector<XVariable *> &list) override;
      virtual void buildConstraintOrdered(string id, vector<XVariable *> &list, OrderType order) override;
      virtual void buildConstraintOrdered(string id, vector<XVariable *> &list, vector<int> &lengths, OrderType order) override;
      virtual void buildConstraintLex(string id, vector<vector<XVariable *>> &lists, OrderType order) override;
      virtual void buildConstraintLexMatrix(string id, vector<vector<XVariable *>> &matrix, OrderType order) override;

      virtual void buildConstraintSum(string id, vector<XVariable *> &list, vector<int> &coeffs, XCondition &cond) override;
      virtual void buildConstraintSum(string id, vector<XVariable *> &list, XCondition &cond) override;
      virtual void buildConstraintSum(string id, vector<XVariable *> &list, vector<XVariable *> &coeffs, XCondition &cond) override;
      virtual void buildConstraintSum(string id, vector<Tree *> &list, vector<int> &coeffs, XCondition &cond) override;
      virtual void buildConstraintSum(string id, vector<Tree *> &list, XCondition &cond) override;
      virtual void buildConstraintAtMost(string id, vector<XVariable *> &list, int value, int k) override;
      virtual void buildConstraintAtLeast(string id, vector<XVariable *> &list, int value, int k) override;
      virtual void buildConstraintExactlyK(string id, vector<XVariable *> &list, int value, int k) override;
      virtual void buildConstraintAmong(string id, vector<XVariable *> &list, vector<int> &values, int k) override;
      virtual void buildConstraintExactlyVariable(string id, vector<XVariable *> &list, int value, XVariable *x) override;
      virtual void buildConstraintCount(string id, vector<XVariable *> &list, vector<int> &values, XCondition &xc) override;
      virtual void buildConstraintCount(string id, vector<XVariable *> &list, vector<XVariable *> &values, XCondition &xc) override;
      virtual void buildConstraintNValues(string id, vector<XVariable *> &list, vector<int> &except, XCondition &xc) override;
      virtual void buildConstraintNValues(string id, vector<XVariable *> &list, XCondition &xc) override;
      virtual void buildConstraintCardinality(string id, vector<XVariable *> &list, vector<int> values, vector<int> &occurs, bool closed) override;
      virtual void buildConstraintCardinality(string id, vector<XVariable *> &list, vector<int> values, vector<XVariable *> &occurs,
                                              bool closed) override;
      virtual void buildConstraintCardinality(string id, vector<XVariable *> &list, vector<int> values, vector<XInterval> &occurs,
                                              bool closed) override;
      virtual void buildConstraintCardinality(string id, vector<XVariable *> &list, vector<XVariable *> values, vector<int> &occurs,
                                              bool closed) override;
      virtual void buildConstraintCardinality(string id, vector<XVariable *> &list, vector<XVariable *> values, vector<XVariable *> &occurs,
                                              bool closed) override;
      virtual void buildConstraintCardinality(string id, vector<XVariable *> &list, vector<XVariable *> values, vector<XInterval> &occurs,
                                              bool closed) override;
      virtual void buildConstraintMinimum(string id, vector<XVariable *> &list, XCondition &xc) override;
      virtual void buildConstraintMinimum(string id, vector<XVariable *> &list, XVariable *index, int startIndex, RankType rank,
                                          XCondition &xc) override;
      virtual void buildConstraintMaximum(string id, vector<XVariable *> &list, XCondition &xc) override;
      virtual void buildConstraintMaximum(string id, vector<XVariable *> &list, XVariable *index, int startIndex, RankType rank,
                                          XCondition &xc) override;

      virtual void buildConstraintElement(string id, vector<XVariable *> &list, int value) override;
      virtual void buildConstraintElement(string id, vector<XVariable *> &list, XVariable *value) override;
      virtual void buildConstraintElement(string id, vector<XVariable *> &list, XVariable *index, int startIndex, XCondition &xc) override;
      virtual void buildConstraintElement(string id, vector<XVariable *> &list, int startIndex, XVariable *index, RankType rank, int value) override;
      virtual void buildConstraintElement(string id, vector<XVariable *> &list, int startIndex, XVariable *index, RankType rank, XVariable *value) override;
      virtual void buildConstraintElement(string id, vector<int> &list, int startIndex, XVariable *index, RankType rank, XVariable *value) override;
      virtual void buildConstraintElement(string id, vector<vector<int> > &matrix, int startRowIndex, XVariable *rowIndex, int startColIndex, XVariable* colIndex, XVariable *value) override;
      virtual void buildConstraintChannel(string id, vector<XVariable *> &list, int startIndex) override;
      virtual void buildConstraintChannel(string id, vector<XVariable *> &list1, int startIndex1, vector<XVariable *> &list2, int startIndex2) override;
      virtual void buildConstraintChannel(string id, vector<XVariable *> &list, int startIndex, XVariable *value) override;
      virtual void buildConstraintStretch(string id, vector<XVariable *> &list, vector<int> &values, vector<XInterval> &widths) override;
      virtual void buildConstraintStretch(string id, vector<XVariable *> &list, vector<int> &values, vector<XInterval> &widths, vector<vector<int>> &patterns) override;
      virtual void buildConstraintNoOverlap(string id, vector<XVariable *> &origins, vector<int> &lengths, bool zeroIgnored) override;
      virtual void buildConstraintNoOverlap(string id, vector<XVariable *> &origins, vector<XVariable *> &lengths, bool zeroIgnored) override;
      virtual void buildConstraintNoOverlap(string id, vector<vector<XVariable *>> &origins, vector<vector<int>> &lengths, bool zeroIgnored) override;
      virtual void buildConstraintNoOverlap(string id, vector<vector<XVariable *>> &origins, vector<vector<XVariable *>> &lengths, bool zeroIgnored) override;
      virtual void buildConstraintCumulative(string id, vector<XVariable *> &origins, vector<int> &lengths, vector<int> &heights, XCondition &xc) override;
      virtual void buildConstraintInstantiation(string id, vector<XVariable *> &list, vector<int> &values) override;
      virtual void buildConstraintClause(string id, vector<XVariable *> &positive, vector<XVariable *> &negative) override ;
      virtual void buildConstraintCircuit(string id, vector<XVariable *> &list, int startIndex) override;
      virtual void buildConstraintCircuit(string id, vector<XVariable *> &list, int startIndex, int size) override;
      virtual void buildConstraintCircuit(string id, vector<XVariable *> &list, int startIndex, XVariable *size) override;
      virtual void buildObjectiveMinimizeExpression(string expr) override;
      virtual void buildObjectiveMaximizeExpression(string expr) override;
      virtual void buildObjectiveMinimizeVariable(XVariable *x) override;
      virtual void buildObjectiveMaximizeVariable(XVariable *x) override;
      virtual void buildObjectiveMinimize(ExpressionObjective type, vector<XVariable *> &list, vector<int> &coefs) override;
      virtual void buildObjectiveMaximize(ExpressionObjective type, vector<XVariable *> &list, vector<int> &coefs) override;
      virtual void buildObjectiveMinimize(ExpressionObjective type, vector<XVariable *> &list) override;
      virtual void buildObjectiveMaximize(ExpressionObjective type, vector<XVariable *> &list) override;
      virtual void buildAnnotationDecision(vector<XVariable*> &list) override;

      bool canonize;
      bool debug = false;
      ::lala::SolverOutput<Allocator>& output;

    private:
      std::vector<F> variables;
      std::vector<F> constraints;
      std::map<std::string, std::vector<std::vector<int>>> extensionAs;
      unsigned int auxiliaryVariables = 0;
      lala::TableDecomposition table_decomposition;

      F make_formula(Node* node);
      lala::Sig to_lala_operator(OrderType operand);
      F to_lala_formula(const XCondition & cond);
      F to_lala_logical_variable(XVariable *&variable);
      F to_lala_logical_variable(const string &variable);
      lala::LVar<Allocator> buildAuxVariableInteger(size_t maxValue);
      lala::LVar<Allocator> buildAuxVariableInteger();

      public:
        size_t num_variables() const {
          return variables.size();
        }

        size_t num_constraints() const {
          return constraints.size();
        }

        battery::shared_ptr<F, Allocator> build_formula() {
          typename F::Sequence seq;
          seq.reserve(variables.size() + constraints.size());
          for(int i = 0; i < variables.size(); ++i) {
            seq.push_back(std::move(variables[i]));
          }
          for(int i = 0; i < constraints.size(); ++i) {
            seq.push_back(std::move(constraints[i]));
          }
          auto f = F::make_nary(lala::AND, std::move(seq));
          return battery::make_shared<F, Allocator>(std::move(f));
        }
    };
}

using namespace XCSP3Core;

template<class T>
void displayList(vector<T> &list, string separator = " ") {
    if(list.size() > 8) {
        for(int i = 0; i < 3; i++)
            cout << list[i] << separator;
        cout << " ... ";
        for(unsigned int i = list.size() - 4; i < list.size(); i++)
            cout << list[i] << separator;
        cout << endl;
        return;
    }
    for(unsigned int i = 0; i < list.size(); i++)
        cout << list[i] << separator;
    cout << endl;
}

void displayList(vector<XVariable *> &list, string separator = " ") {
    if(list.size() > 8) {
        for(int i = 0; i < 3; i++)
            cout << list[i]->id << separator;
        cout << " ... ";
        for(unsigned int i = list.size() - 4; i < list.size(); i++)
            cout << list[i]->id << separator;
        cout << endl;
        return;
    }
    for(unsigned int i = 0; i < list.size(); i++)
        cout << list[i]->id << separator;
    cout << endl;
}



template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::beginInstance(InstanceType type) {
  if(debug) {
    cout << "Start Instance - type=" << type << endl;
  }
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::endInstance() {
  if(debug) {
    cout << "End SAX parsing " << endl;
  }
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::beginVariables() {
  if(debug) {
    cout << " start variables declaration" << endl;
  }
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::endVariables() {
  if(debug) {
    cout << " end variables declaration" << endl << endl;
  }
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::beginVariableArray(string id) {
  if(debug) {
    cout << "    array: " << id << endl;
  }
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::endVariableArray() {}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::beginConstraints() {
  if(debug) {
    cout << " start constraints declaration" << endl;
  }
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::endConstraints() {
  if(debug) {
    cout << "\n end constraints declaration" << endl << endl;
  }
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::beginGroup(string id) {
  if(debug) {
    cout << "   start group of constraint " << id << endl;
  }
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::endGroup() {
  if(debug) {
    cout << "   end group of constraint" << endl;
  }
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::beginBlock(string classes) {
  if(debug) {
    cout << "   start block of constraint classes = " << classes << endl;
  }
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::endBlock() {
  if(debug) {
    cout << "   end block of constraint" << endl;
  }
}

// string id, bool circular
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::beginSlide(string id, bool) {
  if(debug) {
    cout << "   start slide " << id << endl;
  }
  throw std::runtime_error("constraint unsupported");
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::endSlide() {
  if(debug) {
    cout << "   end slide" << endl;
  }
  throw std::runtime_error("constraint unsupported");
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::beginObjectives() {
  if(debug) {
    cout << "   start Objective " << endl;
  }
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::endObjectives() {
  if(debug) {
    cout << "   end Objective " << endl;
  }
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::beginAnnotations() {
  if(debug) {
    cout << "   begin Annotations " << endl;
  }
  throw std::runtime_error("annotations unsupported");
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::endAnnotations() {
  if(debug) {
    cout << "   end Annotations " << endl;
  }
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildVariableInteger(string id, int minValue, int maxValue) {
  if(debug) {
    cout << "    var " << id << " : " << minValue << "..." << maxValue << endl;
  }
  lala::LVar<Allocator> lvar(id.c_str());
  output.add_var(id.c_str());
  variables.push_back(F::make_exists(UNTYPED, lvar, lala::Sort<Allocator>::Int));
  constraints.push_back(F::make_binary(F::make_lvar(UNTYPED, lvar), lala::LEQ, F::make_z(maxValue)));
  constraints.push_back(F::make_binary(F::make_lvar(UNTYPED, lvar), lala::GEQ, F::make_z(minValue)));
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildVariableInteger(string id, vector<int> &values) {
  if(debug) {
    cout << "    var " << id << " : ";
    cout << "        ";
    displayList(values);
  }
  lala::LVar<Allocator> lvar(id.c_str());
  output.add_var(id.c_str());
  variables.push_back(F::make_exists(UNTYPED, lvar, lala::Sort<Allocator>::Int));


  lala::logic_set<F> intervals;
  int start = values[0];
  int end = values[0];

  for (size_t i = 1; i < values.size(); ++i) {
    if (values[i] == end + 1) {
      end = values[i];
    } else {
      auto t = battery::make_tuple(F::make_z(start),F::make_z(end));
      intervals.push_back(t);
      start = values[i];
      end = values[i];
    }
  }
  auto t = battery::make_tuple(F::make_z(start),F::make_z(end));
  intervals.push_back(t);
  constraints.push_back(F::make_binary(F::make_lvar(UNTYPED, lvar), lala::IN, F::make_set(intervals)));
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintExtension(string id, XVariable *variable, vector<int> &tuples, bool support, bool hasStar) {
  if(debug) {
    cout << "\n    extension constraint with one variable: " << id << endl;
    cout << "        " <<(*variable) << " "<< (support ? "support" : "conflict") << " nb tuples: " << tuples.size() << " star: " << hasStar << endl;
    cout << endl;
  }
  typename F::Sequence seq;
  lala::Sig sig = support ? lala::EQ : lala::NEQ;
  for(int i = 0; i < tuples.size(); ++i) {
    if(tuples[i] == INT_MAX && hasStar) {
      return; // constraint satisfied by the presence of a star.
    }
    seq.push_back(
      F::make_binary(F::make_lvar(UNTYPED, lala::LVar<Allocator>(variable->id.c_str())), sig, F::make_z(tuples[i])));
  }
  if(seq.size() == 1) {
    constraints.push_back(seq[0]);
  }
  else if(seq.size() > 1){
    constraints.push_back(F::make_nary(lala::OR, std::move(seq)));
  }
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintExtension(string id, vector<XVariable *> list, vector<vector<int>> &tuples, bool support, bool hasStar) {
  if(debug) {
    cout << "\n    extension constraint : " << id << endl;
    cout << "        " << (support ? "support" : "conflict") << " arity: " << list.size() << " nb tuples: " << tuples.size() << " star: " << hasStar << endl;
    cout << "        ";
    displayList(list);
  }
  extensionAs[id] = tuples;
  lala::Sig sig = support ? lala::EQ : lala::NEQ;
  if(table_decomposition == lala::TableDecomposition::ELEMENTS) {
    size_t numVars = tuples[0].size();
    auto auxVar = F::make_lvar(UNTYPED, buildAuxVariableInteger(numVars - 1));
    for(int i = 0; i < numVars; ++i) {
      for(int j = 0; j < tuples.size(); ++j) {
        // index = i ==> varName = value
        if(!hasStar || tuples[j][i] != INT_MAX) {
          constraints.push_back(F::make_binary(
            F::make_binary(auxVar, lala::EQ,  F::make_z(i)),
            lala::IMPLY,
            F::make_binary(to_lala_logical_variable(list[i]), sig, F::make_z(tuples[j][i]))));
        }
      }
    }
  }
  else if(table_decomposition == lala::TableDecomposition::DISJUNCTIVE) {
    FSeq disjuncts;
    for(int i = 0; i < tuples.size(); ++i) {
      FSeq conjuncts;
      for(int j = 0; j < tuples[i].size(); ++j) {
        // Stars are not added in the conjunction.
        if(!hasStar || tuples[i][j] != INT_MAX) {
          conjuncts.push_back(
            F::make_binary(F::make_lvar(UNTYPED, lala::LVar<Allocator>(list[j]->id.c_str())), sig, F::make_z(tuples[i][j])));
        }
      }
      if(conjuncts.size() == 1) {
        disjuncts.push_back(conjuncts[0]);
      }
      else if(conjuncts.size() > 1) {
        disjuncts.push_back(F::make_nary(lala::AND, std::move(conjuncts)));
      }
    }
    if(disjuncts.size() == 1) {
      constraints.push_back(disjuncts[0]);
    }
    else if(disjuncts.size() > 1){
      constraints.push_back(F::make_nary(lala::OR, std::move(disjuncts)));
    }
  }
  else if(table_decomposition==lala::TableDecomposition::TABLE_PREDICATE) {
    FSeq t_seq;
    t_seq.push_back(F::make_z(tuples.size()));
    t_seq.push_back(F::make_z(list.size()));
    for(int i = 0; i < tuples.size(); ++i) {
      for(int j = 0; j < tuples[i].size(); ++j) {
        if(hasStar && tuples[i][j] == INT_MAX) {
          t_seq.push_back(F::make_lvar(UNTYPED, lala::LVar<Allocator>("*")));
        }
        else {
          t_seq.push_back(F::make_z(tuples[i][j]));
        }
      }
    }
    for(int i = 0; i < list.size(); ++i) {
      t_seq.push_back(F::make_lvar(UNTYPED, lala::LVar<Allocator>(list[i]->id.c_str())));
    }
    constraints.push_back(F::make_nary("tables", std::move(t_seq)));
  }
}

// string id, vector<XVariable *> list, bool support, bool hasStar
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintExtensionAs(string id, vector<XVariable *> list, bool support, bool hasStar) {
  if(debug) {
    cout << "\n    extension constraint similar as previous one: " << id << endl;
  }
  buildConstraintExtension(id, list, extensionAs[id], support, hasStar);
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintIntension(string id, string expr) {
  if(debug) {
    cout << "\n    intension constraint (using string) : " << id << " : " << expr << endl;
  }
  throw std::runtime_error("constraint unsupported (intensions as string)");
}

template<class Allocator>
XCSP3_turbo_callbacks<Allocator>::F XCSP3_turbo_callbacks<Allocator>::make_formula(Node* node) {
  using namespace lala;
  Sig sig;
  switch(node->type) {
    case OUNDEF:
      throw std::runtime_error("found an undefined symbol (`OUNDEF`) which seems to indicate a bug in the XCSP3 model.");
    case ONEG: sig = NEG; break;
    case OABS: sig = ABS; break;
    case OSQR: sig = POW; break;
    case OADD: sig = ADD; break;
    case OSUB: sig = SUB; break;
    case OMUL: sig = MUL; break;
    case ODIV: sig = DIV; break;
    case OMOD: sig = MOD; break;
    case OPOW: sig = POW; break;
    case ODIST: throw std::runtime_error("unsupported DIST"); // return make_dist(node);
    case OMIN: sig = MIN; break;
    case OMAX: sig = MAX; break;
    case OLT: sig = lala::LT; break;
    case OLE: sig = LEQ; break;
    case OGE: sig = GEQ; break;
    case OGT: sig = lala::GT; break;
    case ONE: sig = NEQ; break;
    case OEQ: sig = lala::EQ; break;
    case OSET: throw std::runtime_error("unsupported set"); // return make_set(node);
    case OIN: sig = lala::IN; break;
    case ONOTIN: throw std::runtime_error("unsupported notin");  // return make_notin(node);
    case ONOT: sig = NOT; break;
    case OAND: sig = AND; break;
    case OOR: sig = OR; break;
    case OXOR: sig = XOR; break;
    case OIFF: sig = EQUIV; break;
    case OIMP: sig = IMPLY; break;
    case OIF: sig = ITE; break;
    case OCARD: sig = CARD; break;
    case OUNION: sig = UNION; break;
    case OINTER: sig = INTERSECTION; break;
    case ODIFF:
    case OSDIFF:
    case OHULL:
    case ODJOINT:
      throw std::runtime_error("unsupported set operation");
    case OSUBSET: sig = SUBSET; break;
    case OSUBSEQ: sig = SUBSETEQ; break;
    case OSUPSEQ: sig = SUPSETEQ; break;
    case OSUPSET: sig = SUPSET; break;
    case OCONVEX:
    case OFDIV:
    case OFMOD:
    case OSQRT:
    case ONROOT:
    case OEXP:
    case OLN:
    case OLOG:
    case OSIN:
    case OCOS:
    case OTAN:
    case OASIN:
    case OACOS:
    case OATAN:
    case OSINH:
    case OCOSH:
    case OTANH: throw std::runtime_error("unsupported arithmetic operation");
    case OVAR: return F::make_lvar(UNTYPED, LVar<Allocator>(static_cast<NodeVariable*>(node)->var.c_str()));
    case OPAR:
    case OLONG:
    case ORATIONAL: throw std::runtime_error("unsupported constant");
    case ODECIMAL: return F::make_z(static_cast<NodeConstant*>(node)->val);
    case OSYMBOL: throw std::runtime_error("OSYMBOL should not occur in intension constraint");
  }
  FSeq seq;
  seq.reserve(node->parameters.size());
  for(int i = 0; i < node->parameters.size(); ++i) {
    seq.push_back(make_formula(node->parameters[i]));
  }
  if(node->type == OSQR) {
    seq.push_back(F::make_z(2)); // we represent `x SQR` by `x POW 2`.
  }
  return F::make_nary(sig, std::move(seq));
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintIntension(string id, Tree *tree) {
  if(debug) {
    cout << "\n    intension constraint using canonized tree: " << id << " : ";
    tree->prefixe();
    std::cout << "\n";
  }
  constraints.push_back(make_formula(tree->root));
}

lala::Sig convert(OrderType op) {
  switch(op) {
    case LT: return lala::LT;
    case LE: return lala::LEQ;
    case GT: return lala::GT;
    case GE: return lala::GEQ;
    case EQ: return lala::EQ;
    case NE: return lala::NEQ;
    case IN: return lala::IN;
    default:
      throw std::runtime_error("convert: unsupported operator");
  }
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintPrimitive(string id, OrderType op, XVariable *x, int k, XVariable *y) {
  if(debug) {
    cout << "\n   intension constraint " << id << ": " << x->id << (k >= 0 ? "+" : "") << k << " op " << y->id << endl;
  }
  // push x - y <op> -k
  constraints.push_back(
    F::make_binary(
      F::make_binary(
        F::make_lvar(UNTYPED, lala::LVar<Allocator>(x->id.c_str())),
        lala::SUB,
        F::make_lvar(UNTYPED, lala::LVar<Allocator>(y->id.c_str()))),
      convert(op),
      F::make_z(-k)));
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintPrimitive(string id, OrderType op, XVariable *x, int k) {
  if(debug) {
    cout << "\n   constraint  " << id << ":" << x->id << " op " << k << "\n";
  }
  lala::LVar<Allocator> lvar(x->id.c_str());
  constraints.push_back(F::make_binary(F::make_lvar(UNTYPED, lvar), convert(op), F::make_z(k)));
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintPrimitive(string id, XVariable *x, bool in, int min, int max) {
  if(debug) {
    cout << "\n   constraint  " << id << ":"<< x->id << (in ? " in " : " not in ") << min << ".." << max <<"\n";
  }
  throw std::runtime_error("constraint unsupported (IN set primitive)");
}

// string id, vector<XVariable *> &list, string start, vector<string> &final, vector<XTransition> &transitions
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintRegular(string, vector<XVariable *> &list, string start, vector<string> &final, vector<XTransition> &transitions) {
  if(debug) {
    cout << "\n    regular constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        start: " << start << endl;
    cout << "        final: ";
    displayList(final, ",");
    cout << endl;
    cout << "        transitions: ";
    for(unsigned int i = 0; i < (transitions.size() > 4 ? 4 : transitions.size()); i++) {
        cout << "(" << transitions[i].from << "," << transitions[i].val << "," << transitions[i].to << ") ";
    }
    if(transitions.size() > 4) cout << "...";
    cout << endl;
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<XVariable *> &list, vector<XTransition> &transitions
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintMDD(string, vector<XVariable *> &list, vector<XTransition> &transitions) {
  if(debug) {
    cout << "\n    mdd constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        transitions: ";
    for(unsigned int i = 0; i < (transitions.size() > 4 ? 4 : transitions.size()); i++) {
        cout << "(" << transitions[i].from << "," << transitions[i].val << "," << transitions[i].to << ") ";
    }
    if(transitions.size() > 4) cout << "...";
    cout << endl;
  }
  throw std::runtime_error("constraint unsupported");
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintAlldifferent(string id, vector<XVariable *> &list) {
  if(debug) {
    cout << "\n    allDiff constraint" << id << endl;
    cout << "        ";
    displayList(list);
  }

  for(int i = 0; i < list.size() - 1; ++i) {
    for(int j = i+1; j < list.size(); ++j) {
      constraints.push_back(F::make_binary(
        F::make_lvar(UNTYPED, lala::LVar<Allocator>(list[i]->id.c_str())),
        lala::NEQ,
        F::make_lvar(UNTYPED, lala::LVar<Allocator>(list[j]->id.c_str()))
      ));
    }
  }
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintAlldifferentExcept(string id, vector<XVariable *> &list, vector<int> &except) {
  if(debug) {
    cout << "\n    allDiff constraint with exceptions" << id << endl;
    cout << "        ";
    displayList(list);
    cout << "        Exceptions:";
    displayList(except);
  }
  throw std::runtime_error("constraint unsupported");
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintAlldifferent(string id, vector<Tree *> &list) {
  if(debug) {
    cout << "\n    allDiff constraint with expresions" << id << endl;
    cout << "        ";
    for(Tree *t : list) {
        t->prefixe();std::cout << " ";
    }
    std::cout << std::endl;
  }
  throw std::runtime_error("constraint unsupported");
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintAlldifferentList(string id, vector<vector<XVariable *>> &lists) {
  if(debug) {
    cout << "\n    allDiff list constraint" << id << endl;
    for(unsigned int i = 0; i < (lists.size() < 4 ? lists.size() : 3); i++) {
        cout << "        ";
        displayList(lists[i]);

    }
  }
  throw std::runtime_error("constraint unsupported");
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintAlldifferentMatrix(string id, vector<vector<XVariable *>> &matrix) {
  if(debug) {
    cout << "\n    allDiff matrix constraint" << id << endl;
    for(unsigned int i = 0; i < matrix.size(); i++) {
        cout << "        ";
        displayList(matrix[i]);
    }
  }
  throw std::runtime_error("constraint unsupported");
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintAllEqual(string id, vector<XVariable *> &list) {
  if(debug) {
    cout << "\n    allEqual constraint" << id << endl;
    cout << "        ";
    displayList(list);
  }
  throw std::runtime_error("constraint unsupported");
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintNotAllEqual(string id, vector<XVariable *> &list) {
  if(debug) {
    cout << "\n    not allEqual constraint" << id << endl;
    cout << "        ";
    displayList(list);
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<XVariable *> &list, OrderType order
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintOrdered(string, vector<XVariable *> &list, OrderType order) {
  if(debug) {
    cout << "\n    ordered constraint" << endl;
    string sep;
    if(order == LT) sep = " < ";
    if(order == LE) sep = " <= ";
    if(order == GT) sep = " > ";
    if(order == GE) sep = " >= ";
    cout << "        ";
    displayList(list, sep);
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<XVariable *> &list, vector<int> &lengths, OrderType order
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintOrdered(string, vector<XVariable *> &list, vector<int> &lengths, OrderType order) {
  if(debug) {
    cout << "\n    ordered constraint with lengths" << endl;
    string sep;
    if(order == LT) sep = " < ";
    if(order == LE) sep = " <= ";
    if(order == GT) sep = " > ";
    if(order == GE) sep = " >= ";
    cout << "        ";
    displayList(lengths); cout << "      ";
    displayList(list, sep);
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<vector<XVariable *>> &lists, OrderType order
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintLex(string, vector<vector<XVariable *>> &lists, OrderType order) {
  if(debug) {
    cout << "\n    lex constraint   nb lists: " << lists.size() << endl;
    string sep;
    if(order == LT) sep = " < ";
    if(order == LE) sep = " <= ";
    if(order == GT) sep = " > ";
    if(order == GE) sep = " >= ";
    cout << "        operator: " << sep << endl;
    for(unsigned int i = 0; i < lists.size(); i++) {
        cout << "        list " << i << ": ";
        cout << "        ";
        displayList(lists[i], " ");
    }
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<vector<XVariable *>> &matrix, OrderType order
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintLexMatrix(string, vector<vector<XVariable *>> &matrix, OrderType order) {
  if(debug) {
    cout << "\n    lex matrix constraint   matrix  " << endl;
    string sep;
    if(order == LT) sep = " < ";
    if(order == LE) sep = " <= ";
    if(order == GT) sep = " > ";
    if(order == GE) sep = " >= ";

    for(unsigned int i = 0; i < (matrix.size() < 4 ? matrix.size() : 3); i++) {
        cout << "        ";
        displayList(matrix[i]);
    }
    cout << "        Order " << sep << endl;
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<XVariable *> &list, vector<int> &coeffs, XCondition &cond
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintSum(string, vector<XVariable *> &list, vector<int> &coeffs, XCondition &cond) {
  if(debug) {
    cout << "\n        sum constraint:";
    if(list.size() > 8) {
        for(int i = 0; i < 3; i++)
            cout << (coeffs.size() == 0 ? 1 : coeffs[i]) << "*" << *(list[i]) << " ";
        cout << " ... ";
        for(unsigned int i = list.size() - 4; i < list.size(); i++)
            cout << (coeffs.size() == 0 ? 1 : coeffs[i]) << "*" << *(list[i]) << " ";
    } else {
        for(unsigned int i = 0; i < list.size(); i++)
            cout << (coeffs.size() == 0 ? 1 : coeffs[i]) << "*" << *(list[i]) << " ";
    }
    cout << cond << endl;
  }

  lala::Sig sig = to_lala_operator(cond.op);
  auto right = to_lala_formula(cond);
  FSeq seq;
  for(int i=0;i<list.size();i++) {
    auto var = list[i];
    auto coeff = coeffs[i];
    seq.push_back(F::make_binary(to_lala_logical_variable(var),lala::Sig::MUL,F::make_z(coeff)));
  }
  auto add = F::make_nary(lala::ADD,seq);
  constraints.push_back(F::make_binary(std::move(add),sig,right));

}

// string id, vector<XVariable *> &list, XCondition &cond
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintSum(string, vector<XVariable *> &list, XCondition &cond) {
  if(debug) {
    cout << "\n        unweighted sum constraint:";
    cout << "        ";
    displayList(list, "+");
    cout << cond << endl;
  }
  lala::Sig sig = to_lala_operator(cond.op);
  auto right = to_lala_formula(cond);
  FSeq seq;
  for(auto var : list) {
    seq.push_back(to_lala_logical_variable(var));
  }
  auto add = F::make_nary(lala::ADD,seq);
  constraints.push_back(F::make_binary(std::move(add),sig,right));
}

// string id, vector<XVariable *> &list, vector<XVariable *> &coeffs, XCondition &cond
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintSum(string, vector<XVariable *> &list, vector<XVariable *> &coeffs, XCondition &cond) {
  if(debug) {
    cout << "\n        scalar sum constraint:";
    if(list.size() > 8) {
        for(int i = 0; i < 3; i++)
            cout << coeffs[i]->id << "*" << *(list[i]) << " ";
        cout << " ... ";
        for(unsigned int i = list.size() - 4; i < list.size(); i++)
            cout << coeffs[i]->id << "*" << *(list[i]) << " ";
    } else {
        for(unsigned int i = 0; i < list.size(); i++)
            cout << coeffs[i]->id << "*" << *(list[i]) << " ";
    }
    cout << cond << endl;
  }
  lala::Sig sig = to_lala_operator(cond.op);
  auto right = to_lala_formula(cond);
  FSeq seq;
  for(int i=0;i<list.size();i++) {
    auto var = list[i];
    auto coeff = coeffs[i];
    seq.push_back(F::make_binary(to_lala_logical_variable(var), lala::Sig::MUL, to_lala_logical_variable(coeff)));
  }
  auto add = F::make_nary(lala::ADD,seq);
  constraints.push_back(F::make_binary(std::move(add),sig,right));
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintSum(string id, vector<Tree *> &list, vector<int> &coeffs, XCondition &cond) {
  if(debug) {
    std::cout << "\n        sum with expression constraint;";
    if(list.size() > 8) {
        for(int i = 0; i < 3; i++) {
            cout << coeffs[i];
            list[i]->prefixe();
        }
        cout << " ... ";
        for(unsigned int i = list.size() - 4; i < list.size(); i++) {
            cout << coeffs[i];
            list[i]->prefixe();
        }
    } else {
        for(unsigned int i = 0; i < list.size(); i++) {
            cout << coeffs[i];
            list[i]->prefixe();
        }
    }
    cout << cond << endl;
  }
  throw std::runtime_error("constraint unsupported");
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintSum(string id, vector<Tree *> &list, XCondition &cond) {
  if(debug) {
    if(list.size() > 8) {
        for(int i = 0; i < 3; i++) {
            list[i]->prefixe();
        }
        cout << " ... ";
        for(unsigned int i = list.size() - 4; i < list.size(); i++) {
            list[i]->prefixe();
        }
    } else {
        for(unsigned int i = 0; i < list.size(); i++) {
            list[i]->prefixe();
        }
    }
    cout << cond << endl;
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, int value, int k
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintAtMost(string, vector<XVariable *> &list, int value, int k) {
  if(debug) {
    cout << "\n    AtMost constraint: val=" << value << " k=" << k << endl;
    cout << "        ";
    displayList(list);
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<XVariable *> &list, int value, int k
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintAtLeast(string, vector<XVariable *> &list, int value, int k) {
  if(debug) {
    cout << "\n    Atleast constraint: val=" << value << " k=" << k << endl;
    cout << "        ";
    displayList(list);
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<XVariable *> &list, int value, int k
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintExactlyK(string, vector<XVariable *> &list, int value, int k) {
  if(debug) {
    cout << "\n    Exactly constraint: val=" << value << " k=" << k << endl;
    cout << "        ";
    displayList(list);
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<XVariable *> &list, vector<int> &values, int k
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintAmong(string, vector<XVariable *> &list, vector<int> &values, int k) {
  if(debug) {
    cout << "\n    Among constraint: k=" << k << endl;
    cout << "        ";
    displayList(list);
    cout << "        values:";
    displayList(values);
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<XVariable *> &list, int value, XVariable *x
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintExactlyVariable(string, vector<XVariable *> &list, int value, XVariable *x) {
  if(debug) {
    cout << "\n    Exactly Variable constraint: val=" << value << " variable=" << *x << endl;
    cout << "        ";
    displayList(list);
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<XVariable *> &list, vector<int> &values, XCondition &xc
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintCount(string, vector<XVariable *> &list, vector<int> &values, XCondition &xc) {
  if(debug) {
    cout << "\n    count constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        values: ";
    cout << "        ";
    displayList(values);
    cout << "        condition: " << xc << endl;
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<XVariable *> &list, vector<XVariable *> &values, XCondition &xc
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintCount(string, vector<XVariable *> &list, vector<XVariable *> &values, XCondition &xc) {
  if(debug) {
    cout << "\n    count constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        values: ";
    displayList(values);
    cout << "        condition: " << xc << endl;
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<XVariable *> &list, vector<int> &except, XCondition &xc
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintNValues(string, vector<XVariable *> &list, vector<int> &except, XCondition &xc) {
  if(debug) {
    cout << "\n    NValues with exceptions constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        exceptions: ";
    displayList(except);
    cout << "        condition:" << xc << endl;
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<XVariable *> &list, XCondition &xc
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintNValues(string, vector<XVariable *> &list, XCondition &xc) {
  if(debug) {
    cout << "\n    NValues  constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        condition:" << xc << endl;
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<XVariable *> &list, vector<int> values, vector<int> &occurs, bool closed
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintCardinality(string, vector<XVariable *> &list, vector<int> values, vector<int> &occurs, bool closed) {
  if(debug) {
    cout << "\n    Cardinality constraint (int values, int occurs)  constraint closed: " << closed << endl;
    cout << "        ";
    displayList(list);
    cout << "        values:";
    displayList(values);
    cout << "        occurs:";
    displayList(occurs);
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<XVariable *> &list, vector<int> values, vector<XVariable *> &occurs, bool closed
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintCardinality(string, vector<XVariable *> &list, vector<int> values, vector<XVariable *> &occurs, bool closed) {
  if(debug) {
    cout << "\n    Cardinality constraint (int values, var occurs)  constraint closed: " << closed << endl;
    cout << "        ";
    displayList(list);
    cout << "        values:";
    displayList(values);
    cout << "        occurs:";
    displayList(occurs);
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<XVariable *> &list, vector<int> values, vector<XInterval> &occurs, bool closed
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintCardinality(string, vector<XVariable *> &list, vector<int> values, vector<XInterval> &occurs, bool closed) {
  if(debug) {
    cout << "\n    Cardinality constraint (int values, interval occurs)  constraint closed: " << closed << endl;
    cout << "        ";
    displayList(list);
    cout << "        values:";
    displayList(values);
    cout << "        occurs:";
    displayList(occurs);
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<XVariable *> &list, vector<XVariable *> values, vector<int> &occurs, bool closed
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintCardinality(string, vector<XVariable *> &list, vector<XVariable *> values, vector<int> &occurs, bool closed) {
  if(debug) {
    cout << "\n    Cardinality constraint (var values, int occurs)  constraint closed: " << closed << endl;
    cout << "        ";
    displayList(list);
    cout << "        values:";
    displayList(values);
    cout << "        occurs:";
    displayList(occurs);
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<XVariable *> &list, vector<XVariable *> values, vector<XVariable *> &occurs, bool closed
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintCardinality(string, vector<XVariable *> &list, vector<XVariable *> values, vector<XVariable *> &occurs, bool closed) {
  if(debug) {
    cout << "\n    Cardinality constraint (var values, var occurs)  constraint closed: " << closed << endl;
    cout << "        ";
    displayList(list);
    cout << "        values:";
    displayList(values);
    cout << "        occurs:";
    displayList(occurs);
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<XVariable *> &list, vector<XVariable *> values, vector<XInterval> &occurs, bool closed
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintCardinality(string, vector<XVariable *> &list, vector<XVariable *> values, vector<XInterval> &occurs, bool closed) {
  if(debug) {
    cout << "\n    Cardinality constraint (var values, interval occurs)  constraint closed: " << closed << endl;
    cout << "        ";
    displayList(list);
    cout << "        values:";
    displayList(values);
    cout << "        occurs:";
    displayList(occurs);
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<XVariable *> &list, XCondition &xc
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintMinimum(string, vector<XVariable *> &list, XCondition &xc) {
  if(debug) {
    cout << "\n    minimum  constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        condition: " << xc << endl;
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<XVariable *> &list, XVariable *index, int startIndex, RankType rank, XCondition &xc
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintMinimum(string, vector<XVariable *> &list, XVariable *index, int startIndex, RankType, XCondition &xc) {
  if(debug) {
    cout << "\n    arg_minimum  constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        index:" << *index << endl;
    cout << "        Start index : " << startIndex << endl;
    cout << "        condition: " << xc << endl;
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<XVariable *> &list, XCondition &xc
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintMaximum(string, vector<XVariable *> &list, XCondition &xc) {
  if(debug) {
    cout << "\n    maximum  constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        condition: " << xc << endl;
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<XVariable *> &list, XVariable *index, int startIndex, RankType rank, XCondition &xc
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintMaximum(string, vector<XVariable *> &list, XVariable *index, int startIndex, RankType, XCondition &xc) {
  if(debug) {
    cout << "\n    arg_maximum  constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        index:" << *index << endl;
    cout << "        Start index : " << startIndex << endl;
    cout << "        condition: " << xc << endl;
  }
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, int value
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintElement(string, vector<XVariable *> &list, int value) {
  if(debug) {
    cout << "\n    element constant constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        value: " << value << endl;
  }
  auto v = F::make_z(value);
  FSeq seq;
  for(auto & variable : list) {
    seq.push_back(F::make_binary(to_lala_logical_variable(variable), lala::EQ, v));
  }
  constraints.push_back(F::make_nary(lala::OR, std::move(seq)));

}

// string id, vector<XVariable *> &list, XVariable *value
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintElement(string, vector<XVariable *> &list, XVariable *value) {
  if(debug) {
    cout << "\n    element variable constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        value: " << *value << endl;
  }
  auto v = to_lala_logical_variable(value);
  FSeq seq;
  for(auto & variable : list) {
    seq.push_back(F::make_binary(to_lala_logical_variable(variable), lala::EQ,  v));
  }
  constraints.push_back(F::make_nary(lala::OR, std::move(seq)));
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintElement(string id, vector<XVariable *> &list, XVariable *index, int startIndex, XCondition &xc) {
  for(int i = 0; i < list.size(); ++i) {
    // index = (i+1) ==> varName = value
    constraints.push_back(F::make_binary(
      F::make_binary(to_lala_logical_variable(index), lala::EQ,  F::make_z(i)),
      lala::IMPLY,
      F::make_binary(to_lala_logical_variable(list[i]), to_lala_operator(xc.op), to_lala_formula(xc))));
  }
}

// string id, vector<XVariable *> &list, int startIndex, XVariable *index, RankType rank, int value
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintElement(string id, vector<XVariable *> &list, int startIndex, XVariable *index, RankType, int value) {
  if(debug) {
    cout << "\n    element constant (with index) constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        value: " << value << endl;
    cout << "        Start index : " << startIndex << endl;
    cout << "        index : " << *index << endl;
  }
  for(int i = 0; i < list.size(); ++i) {
    // index = (i+1) ==> varName = value
    constraints.push_back(F::make_binary(
      F::make_binary(to_lala_logical_variable(index), lala::EQ,  F::make_z(i)),
      lala::IMPLY,
      F::make_binary(to_lala_logical_variable(list[i]), lala::EQ,  F::make_z(value))));
  }
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintElement(string id, vector<vector<int> > &matrix, int startRowIndex, XVariable *rowIndex, int startColIndex, XVariable* colIndex, XVariable *value) {
  if(debug) {
    cout << "\n    element matrix with rowIndex, colIndex and Value variables\n";
    for(unsigned int i = 0; i < matrix.size(); i++) {
        for(int j = 0; j < matrix.size(); j++)
            cout << matrix[i][j] << " ";
        cout << endl;
    }
    cout << "        row index : " << *rowIndex << endl;
    cout << "        col index : " << *colIndex << endl;
    cout << "        value     : " << *value << endl;
  }

  for(int i = 0; i < matrix.size(); ++i) {
    for(int j=0;j<matrix[i].size();j++) {
      // index = (i+1) ==> varName = value
      auto leftAnd = F::make_binary(to_lala_logical_variable(rowIndex), lala::EQ,  F::make_z(i));
      auto rightAnd = F::make_binary(to_lala_logical_variable(colIndex), lala::EQ,  F::make_z(j));
      auto andand = F::make_binary(leftAnd,lala::AND,rightAnd);
      constraints.push_back(F::make_binary(
        andand,
        lala::IMPLY,
        F::make_binary(F::make_z(matrix[i][j]), lala::EQ,  to_lala_logical_variable(value))));
    }
  }
}

// string id, vector<XVariable *> &list, int startIndex, XVariable *index, RankType rank, XVariable *value
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintElement(string, vector<XVariable *> &list, int startIndex, XVariable *index, RankType, XVariable *value) {
  if(debug) {
    cout << "\n    element variable (with index) constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        value: " << *value << endl;
    cout << "        Start index : " << startIndex << endl;
    cout << "        index : " << *index << endl;
  }
  for(int i = 0; i < list.size(); ++i) {
    // index = (i+1) ==> varName = value
    constraints.push_back(F::make_binary(
      F::make_binary(to_lala_logical_variable(index), lala::EQ,  F::make_z(i)),
      lala::IMPLY,
      F::make_binary(to_lala_logical_variable(list[i]), lala::EQ,  to_lala_logical_variable(value))));
  }
}

// string, vector<int> &list, int startIndex, XVariable *index, RankType rank, XVariable *value
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintElement(string, vector<int> &list, int startIndex, XVariable *index, RankType, XVariable *value) {
  if(debug) {
    cout << "\n    element variable with list of integers (with index) constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        value: " << *value << endl;
    cout << "        Start index : " << startIndex << endl;
    cout << "        index : " << *index << endl;
  }
  for(int i = 0; i < list.size(); ++i) {
    // index = (i+1) ==> varName = value
    constraints.push_back(F::make_binary(
      F::make_binary(to_lala_logical_variable(index), lala::EQ,  F::make_z(i)),
      lala::IMPLY,
      F::make_binary(F::make_z(list[i]), lala::EQ, to_lala_logical_variable(value))));
  }
}

// string id, vector<XVariable *> &list, int startIndex
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintChannel(string, vector<XVariable *> &list, int startIndex) {
  if(debug) {
    cout << "\n    channel constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        Start index : " << startIndex << endl;
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<XVariable *> &list1, int startIndex1, vector<XVariable *> &list2, int startIndex2
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintChannel(string, vector<XVariable *> &list1, int, vector<XVariable *> &list2, int) {
  if(debug) {
    cout << "\n    channel constraint" << endl;
    cout << "        list1 ";
    displayList(list1);
    cout << "        list2 ";
    displayList(list2);
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<XVariable *> &list, int startIndex, XVariable *value
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintChannel(string, vector<XVariable *> &list, int, XVariable *value) {
  if(debug) {
    cout << "\n    channel constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        value: " << *value << endl;
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<XVariable *> &list, vector<int> &values, vector<XInterval> &widths
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintStretch(string, vector<XVariable *> &list, vector<int> &values, vector<XInterval> &widths) {
  if(debug) {
    cout << "\n    stretch constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        values :";
    displayList(values);
    cout << "        widths:";
    displayList(widths);
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<XVariable *> &list, vector<int> &values, vector<XInterval> &widths, vector<vector<int>> &patterns
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintStretch(string, vector<XVariable *> &list, vector<int> &values, vector<XInterval> &widths, vector<vector<int>> &patterns) {
  if(debug) {
    cout << "\n    stretch constraint (with patterns)" << endl;
    cout << "        ";
    displayList(list);
    cout << "        values :";
    displayList(values);
    cout << "        widths:";
    displayList(widths);
    cout << "        patterns";
    for(unsigned int i = 0; i < patterns.size(); i++)
        cout << "(" << patterns[i][0] << "," << patterns[i][1] << ") ";
    cout << endl;
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<XVariable *> &origins, vector<int> &lengths, bool zeroIgnored
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintNoOverlap(string, vector<XVariable *> &origins, vector<int> &lengths, bool) {
  if(debug) {
    cout << "\n    nooverlap constraint" << endl;
    cout << "        origins";
    displayList(origins);
    cout << "        lengths";
    displayList(lengths);
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<XVariable *> &origins, vector<XVariable *> &lengths, bool zeroIgnored
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintNoOverlap(string, vector<XVariable *> &origins, vector<XVariable *> &lengths, bool) {
  if(debug) {
    cout << "\n    nooverlap constraint" << endl;
    cout << "        origins:";
    displayList(origins);
    cout << "        lengths";
    displayList(lengths);
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<vector<XVariable *>> &origins, vector<vector<int>> &lengths, bool zeroIgnored
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintNoOverlap(string, vector<vector<XVariable *>> &origins, vector<vector<int>> &lengths, bool) {
  if(debug) {
    cout << "\n    kdim (int lengths) nooverlap constraint" << endl;
    cout << "origins: " << endl;
    for(unsigned int i = 0; i < origins.size(); i++) {
        cout << "        ";
        displayList(origins[i]);
    }
    cout << "lengths: " << endl;
    for(unsigned int i = 0; i < origins.size(); i++) {
        cout << "        ";
        displayList(lengths[i]);
    }
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<vector<XVariable *>> &origins, vector<vector<XVariable *>> &lengths, bool zeroIgnored
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintNoOverlap(string, vector<vector<XVariable *>> &origins, vector<vector<XVariable *>> &lengths, bool) {
  if(debug) {
    cout << "\n    kdim (lenghts vars nooverlap constraint" << endl;
    cout << "origins: " << endl;
    for(unsigned int i = 0; i < origins.size(); i++) {
        cout << "        ";
        displayList(origins[i]);
    }
    cout << "lengths: " << endl;
    for(unsigned int i = 0; i < origins.size(); i++) {
        cout << "        ";
        displayList(lengths[i]);
    }
  }
  throw std::runtime_error("constraint unsupported");
}

/**
  * The callback function related to a cumulative constraint with variable origins, int lengths and int heights
  * See http://xcsp.org/specifications/cumulative
  *
  * Example:
  * <cumulative>
  *     <origins> s1 s2 s3 s4 </origins>
  *     <lengths> 1 2 3 4 </lengths>
  *     <heights> 3 4 5 6 </heights>
  *     <condition> (le,4) </condition>
  * </cumulative>
  *
  * @param id the id (name) of the constraint
  * @param origins the vector of origins
  * @param lengths the vector of lenghts (here ints)
  * @param heights the vector of heights (here ints)
  * @param xc the condition (see XCondition)
  */
// string id, vector<vector<XV
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintCumulative(string id, vector<XVariable *> &origins, vector<int> &lengths, vector<int> &heights, XCondition &xc) {
  battery::vector<F> vars;
  for(int j=0;j<origins.size();j++) {
    for(int i=0;i<origins.size();i++) {
      if(i == j) {
        continue;
      }
      auto left = F::make_binary(to_lala_logical_variable(origins[i]), lala::LEQ, to_lala_logical_variable(origins[j]));
      auto right = F::make_binary(to_lala_logical_variable(origins[j]), lala::LT, F::make_binary(to_lala_logical_variable(origins[i]), lala::ADD, F::make_z(lengths[i])));
      auto andand = F::make_binary(left, lala::AND, right);
      auto bij = F::make_lvar(UNTYPED, buildAuxVariableInteger(1));
      vars.push_back(bij);
      auto equiv = F::make_binary(bij,lala::EQUIV,andand);
      constraints.push_back(equiv);
    }
  }

  for(int j=0;j<origins.size();j++) {
    auto rkj = F::make_z(heights[j]);
    FSeq sum;
    sum.push_back(rkj);
    for(int i=0;i<origins.size();i++) {
      if(i == j) {
        continue;
      }
      auto index = i*(origins.size()-1)+j;
      sum.push_back(F::make_binary(F::make_z(heights[i]),lala::MUL,vars[index]));
    }
    auto left = F::make_nary(lala::ADD,std::move(sum));
    constraints.push_back(F::make_binary(left,to_lala_operator(xc.op),to_lala_formula(xc)));
  }

}




// string id, vector<XVariable *> &list, vector<int> &values
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintInstantiation(string, vector<XVariable *> &list, vector<int> &values) {
  if(debug) {
    cout << "\n    instantiation constraint" << endl;
    cout << "        list:";
    displayList(list);
    cout << "        values:";
    displayList(values);
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<XVariable *> &list, vector<int> &values
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintClause(string, vector<XVariable *> &positive, vector<XVariable *> &negative) {
  if(debug) {
    cout << "\n    Clause constraint" << endl;
    cout << "        positive lits size:" << positive.size() <<" ";
    displayList(positive);
    cout << "        negative lits size:" << negative.size() <<" ";
    displayList(negative);
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<XVariable *> &list, int startIndex
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintCircuit(string, vector<XVariable *> &list, int startIndex) {
  if(debug) {
    cout << "\n    circuit constraint" << endl;
    cout << "        list:";
    displayList(list);
    cout << "        startIndex:" << startIndex << endl;
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<XVariable *> &list, int startIndex, int size
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintCircuit(string, vector<XVariable *> &list, int startIndex, int size) {
  if(debug) {
    cout << "\n    circuit constraint" << endl;
    cout << "        list:";
    displayList(list);
    cout << "        startIndex:" << startIndex << endl;
    cout << "        size: " << size << endl;
  }
  throw std::runtime_error("constraint unsupported");
}

// string id, vector<XVariable *> &list, int startIndex, XVariable *size
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintCircuit(string, vector<XVariable *> &list, int startIndex, XVariable *size) {
  if(debug) {
    cout << "\n    circuit constraint" << endl;
    cout << "        list:";
    displayList(list);
    cout << "        startIndex:" << startIndex << endl;
    cout << "        size: " << size->id << endl;
  }
  throw std::runtime_error("constraint unsupported");
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildObjectiveMinimizeExpression(string expr) {
  if(debug) {
    cout << "\n    objective: minimize" << expr << endl;
  }
  throw std::runtime_error("constraint unsupported");
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildObjectiveMaximizeExpression(string expr) {
  if(debug) {
    cout << "\n    objective: maximize" << expr << endl;
  }
  throw std::runtime_error("constraint unsupported");
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildObjectiveMinimizeVariable(XVariable *x) {
  if(debug) {
    cout << "\n    objective: minimize variable " << x << endl;
  }
  constraints.push_back(F::make_unary(lala::MINIMIZE, F::make_lvar(UNTYPED, x->id.c_str())));
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildObjectiveMaximizeVariable(XVariable *x) {
  if(debug) {
    cout << "\n    objective: maximize variable " << x << endl;
  }
  constraints.push_back(F::make_unary(lala::MAXIMIZE, F::make_lvar(UNTYPED, x->id.c_str())));
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildObjectiveMinimize(ExpressionObjective type, vector<XVariable *> &list, vector<int> &coefs) {
  if(debug) {
    cout<<"objective of type " << type << " " << endl;
    displayList(list);
    displayList(coefs);
  }
  if(type!=SUM_O) {
    throw std::runtime_error("minimization of type "+to_string(type)+" unsupported");
  }

  battery::vector<F> sequences;
  auto optVariable = F::make_lvar(UNTYPED, buildAuxVariableInteger());
  for(int i=0;i<list.size();i++) {
    sequences.push_back(F::make_binary(F::make_lvar(UNTYPED,list[i]->id),lala::MUL,F::make_z(coefs[i])));
  }

  constraints.push_back(F::make_binary(optVariable,lala::EQ,F::make_nary(lala::ADD,sequences)));
  constraints.push_back(F::make_unary(lala::MINIMIZE,optVariable));
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildObjectiveMaximize(ExpressionObjective type, vector<XVariable *> &list, vector<int> &coefs) {
  if(debug) {
    cout<<"objective of type " << type << " " << endl;
    displayList(list);
    displayList(coefs);
  }
  if(type!=SUM_O) {
    throw std::runtime_error("maximization of type "+to_string(type)+" unsupported");
  }

  battery::vector<F> sequences;
  auto optVariable = F::make_lvar(UNTYPED, buildAuxVariableInteger());

  for(int i=0;i<list.size();i++) {
    sequences.push_back(F::make_binary(F::make_lvar(UNTYPED,list[i]->id),lala::MUL,F::make_z(coefs[i])));
  }
  constraints.push_back(F::make_binary(optVariable,lala::EQ,F::make_nary(lala::ADD,sequences)));
  constraints.push_back(F::make_unary(lala::MAXIMIZE,optVariable));
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildObjectiveMinimize(ExpressionObjective type, vector<XVariable *> &list) {
  if(debug) {
    cout<<"objective of type " << type << " " << endl;
    displayList(list);
  }
  if(type!=SUM_O) {
    throw std::runtime_error("minimization of type "+to_string(type)+" unsupported");
  }

  battery::vector<F> sequences;
  auto optVariable = F::make_lvar(UNTYPED, buildAuxVariableInteger());
  for(int i=0;i<list.size();i++) {
    sequences.push_back(F::make_lvar(UNTYPED,list[i]->id));
  }
  constraints.push_back(F::make_binary(optVariable,lala::EQ,F::make_nary(lala::ADD,sequences)));
  constraints.push_back(F::make_unary(lala::MINIMIZE,optVariable));
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildObjectiveMaximize(ExpressionObjective type, vector<XVariable *> &list) {
  if(debug) {
    cout<<"objective of type " << type << " " << endl;
    displayList(list);
  }
  if(type!=SUM_O) {
    throw std::runtime_error("maximization of type "+to_string(type)+" unsupported");
  }

  battery::vector<F> sequences;
  auto optVariable = F::make_lvar(UNTYPED, buildAuxVariableInteger());
  for(int i=0;i<list.size();i++) {
    sequences.push_back(F::make_lvar(UNTYPED,list[i]->id));
  }
  constraints.push_back(F::make_binary(optVariable,lala::EQ,F::make_nary(lala::ADD,sequences)));
  constraints.push_back(F::make_unary(lala::MAXIMIZE,optVariable));
}


template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildAnnotationDecision(vector<XVariable*> &list) {
  if(debug) {
    std::cout << "       decision variables" << std::endl<< "       ";
    displayList(list);
  }
  throw std::runtime_error("constraint unsupported");
}

template<class Allocator>
lala::Sig XCSP3_turbo_callbacks<Allocator>::to_lala_operator(OrderType op) {
  switch (op) {
    case LE: return lala::Sig::LEQ;
    case LT: return lala::Sig::LT;
    case GE: return lala::Sig::GEQ;
    case GT: return lala::Sig::GT;
    case IN: return lala::Sig::IN;
    case NE: return lala::Sig::NEQ;
    case EQ: return lala::Sig::EQ;
    default:
      throw std::runtime_error(op+" is not a supported operator type.");
  }
}

template<class Allocator>
XCSP3_turbo_callbacks<Allocator>::F XCSP3_turbo_callbacks<Allocator>::to_lala_formula(const XCondition & cond) {
  switch (cond.operandType) {
    case INTEGER: return F::make_z(cond.val);
    case VARIABLE: return to_lala_logical_variable(cond.var);
    default:
      throw std::runtime_error(cond.operandType+" is not a supported operand type.");
  }
}

template<class Allocator>
XCSP3_turbo_callbacks<Allocator>::F XCSP3_turbo_callbacks<Allocator>::to_lala_logical_variable(XVariable *&variable) {
  return F::make_lvar(UNTYPED, lala::LVar<Allocator>(variable->id.c_str()));
}

template<class Allocator>
XCSP3_turbo_callbacks<Allocator>::F XCSP3_turbo_callbacks<Allocator>::to_lala_logical_variable(const string &variable) {
  return F::make_lvar(UNTYPED, lala::LVar<Allocator>(variable.c_str()));
}

template <class Allocator>
lala::LVar<Allocator> XCSP3_turbo_callbacks<Allocator>::buildAuxVariableInteger(size_t maxValue) {
  lala::LVar<Allocator> auxVar("aux_"+::std::to_string(auxiliaryVariables));
  auxiliaryVariables++;
  variables.push_back(F::make_exists(UNTYPED, auxVar, lala::Sort<Allocator>::Int));
  constraints.push_back(F::make_binary(F::make_lvar(UNTYPED, auxVar), lala::LEQ, F::make_z(maxValue)));
  constraints.push_back(F::make_binary(F::make_lvar(UNTYPED, auxVar), lala::GEQ, F::make_z(0)));
  return auxVar;
}

template <class Allocator>
lala::LVar<Allocator> XCSP3_turbo_callbacks<Allocator>::buildAuxVariableInteger() {
  lala::LVar<Allocator> auxVar("aux_"+::std::to_string(auxiliaryVariables));
  auxiliaryVariables++;
  variables.push_back(F::make_exists(UNTYPED, auxVar, lala::Sort<Allocator>::Int));
  return auxVar;
}


#endif // XCSP3_PARSER_HPP
