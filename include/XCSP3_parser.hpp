// Copyright 2022 Pierre Talbot

#ifndef  XCSP3_PARSER_HPP
#define  XCSP3_PARSER_HPP

#include <iostream>
#include <optional>

#include "XCSP3Tree.h"
#include "XCSP3CoreCallbacks.h"
#include "XCSP3Variable.h"
#include "XCSP3CoreParser.h"

#include "ast.hpp"

namespace XCSP3Core {
  template <class Allocator>
  class XCSP3_turbo_callbacks;
}

namespace lala {
  template<class Allocator>
  SFormula<Allocator> parse_xcsp3(const std::string& filename, AType store_ty, AType ipc_ty) {
    XCSP3Core::XCSP3_turbo_callbacks<Allocator> cb(store_ty, ipc_ty);
    XCSP3Core::XCSP3CoreParser parser(&cb);
    parser.parse(filename.c_str());
    return cb.build_formula();
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
    public:
        using XCSP3CoreCallbacks::buildConstraintMinimum;
        using XCSP3CoreCallbacks::buildConstraintMaximum;
        using XCSP3CoreCallbacks::buildConstraintElement;
        using XCSP3CoreCallbacks::buildObjectiveMinimize;
        using XCSP3CoreCallbacks::buildObjectiveMaximize;

        XCSP3_turbo_callbacks(lala::AType store_ty, lala::AType ipc_ty):
          XCSP3CoreCallbacks(), canonize(true), store_ty(store_ty), ipc_ty(ipc_ty) {}

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

        using F = lala::TFormula<Allocator>;
        using SF = lala::SFormula<Allocator>;

      private:
        lala::AType store_ty;
        lala::AType ipc_ty;

        std::vector<F> variables;
        std::vector<F> constraints;
        std::optional<lala::LVar<Allocator>> minimize;
        std::optional<lala::LVar<Allocator>> maximize;

        F make_formula(Node* node);

      public:

        SF build_formula() {
          typename F::Sequence seq;
          seq.reserve(variables.size() + constraints.size());
          for(int i = 0; i < variables.size(); ++i) {
            seq.push_back(std::move(variables[i]));
          }
          for(int i = 0; i < constraints.size(); ++i) {
            seq.push_back(std::move(constraints[i]));
          }
          auto f = F::make_nary(lala::AND, std::move(seq), ipc_ty);
          if(minimize.has_value() && maximize.has_value()) {
            throw std::runtime_error("Multiple objectives are unsupported.");
          }
          if(minimize.has_value()) {
            return SF(std::move(f), SF::MINIMIZE, *minimize);
          }
          else if(maximize.has_value()) {
            return SF(std::move(f), SF::MAXIMIZE, *maximize);
          }
          else {
            return SF(std::move(f));
          }
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
  variables.push_back(F::make_exists(store_ty, lvar, lala::Int));
  constraints.push_back(F::make_binary(F::make_lvar(store_ty, lvar), lala::LEQ, F::make_z(maxValue), store_ty));
  constraints.push_back(F::make_binary(F::make_lvar(store_ty, lvar), lala::GEQ, F::make_z(minValue), store_ty));
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildVariableInteger(string id, vector<int> &values) {
  if(debug) {
    cout << "    var " << id << " : ";
    cout << "        ";
    displayList(values);
  }
  throw std::runtime_error("set variable unsupported");
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintExtension(string id, vector<XVariable *> list, vector<vector<int>> &tuples, bool support, bool hasStar) {
  if(debug) {
    cout << "\n    extension constraint : " << id << endl;
    cout << "        " << (support ? "support" : "conflict") << " arity: " << list.size() << " nb tuples: " << tuples.size() << " star: " << hasStar << endl;
    cout << "        ";
    displayList(list);
  }
  throw std::runtime_error("constraint unsupported (extension)");
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintExtension(string id, XVariable *variable, vector<int> &tuples, bool support, bool hasStar) {
  if(debug) {
    cout << "\n    extension constraint with one variable: " << id << endl;
    cout << "        " <<(*variable) << " "<< (support ? "support" : "conflict") << " nb tuples: " << tuples.size() << " star: " << hasStar << endl;
    cout << endl;
  }
  throw std::runtime_error("constraint unsupported (extension)");
}

// string id, vector<XVariable *> list, bool support, bool hasStar
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintExtensionAs(string id, vector<XVariable *>, bool, bool) {
  if(debug) {
    cout << "\n    extension constraint similar as previous one: " << id << endl;
  }
  throw std::runtime_error("constraint unsupported (extensionAs)");
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
    case OSQR: sig = SQR; break;
    case OADD: sig = ADD; break;
    case OSUB: sig = SUB; break;
    case OMUL: sig = MUL; break;
    case ODIV: sig = DIV; break;
    case OMOD: sig = MOD; break;
    case OPOW: sig = POW; break;
    case ODIST: throw std::runtime_error("unsupported DIST"); // return make_dist(node);
    case OMIN: sig = MEET; break;
    case OMAX: sig = JOIN; break;
    case OLT: sig = lala::LT; break;
    case OLE: sig = LEQ; break;
    case OGE: sig = GEQ; break;
    case OGT: sig = lala::GT; break;
    case ONE: sig = NEQ; break;
    case OEQ: sig = lala::EQ; break;
    case OSET:  throw std::runtime_error("unsupported set"); // return make_set(node);
    case OIN: sig = GEQ; break;
    case ONOTIN: throw std::runtime_error("unsupported notin");  // return make_notin(node);
    case ONOT: sig = NOT; break;
    case OAND: sig = AND; break;
    case OOR: sig = OR; break;
    case OXOR: sig = XOR; break;
    case OIFF: sig = EQUIV; break;
    case OIMP: sig = IMPLY; break;
    case OIF: throw std::runtime_error("unsupported if"); // return make_if(node);
    case OCARD: sig = CARD; break;
    case OUNION: sig = MEET; break;
    case OINTER: sig = JOIN; break;
    case ODIFF:
    case OSDIFF:
    case OHULL:
    case ODJOINT:
      throw std::runtime_error("unsupported set operation");
    case OSUBSET: sig = lala::GT; break;
    case OSUBSEQ: sig = GEQ; break;
    case OSUPSEQ: sig = LEQ; break;
    case OSUPSET: sig = lala::LT; break;
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
    case OVAR: return F::make_lvar(store_ty, LVar<Allocator>(static_cast<NodeVariable*>(node)->var.c_str()));
    case OPAR:
    case OLONG:
    case ORATIONAL: throw std::runtime_error("unsupported constant");
    case ODECIMAL: return F::make_z(static_cast<NodeConstant*>(node)->val);
    case OSYMBOL: throw std::runtime_error("OSYMBOL should not occur in intension constraint");
  }
  typename F::Sequence seq;
  seq.reserve(node->parameters.size());
  for(int i = 0; i < node->parameters.size(); ++i) {
    seq.push_back(make_formula(node->parameters[i]));
  }
  return F::make_nary(sig, std::move(seq), ipc_ty);
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
    case IN: return lala::GEQ;
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
        F::make_lvar(store_ty, lala::LVar<Allocator>(x->id.c_str())),
        lala::SUB,
        F::make_lvar(store_ty, lala::LVar<Allocator>(y->id.c_str())),
        ipc_ty),
      convert(op),
      F::make_z(-k)));
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintPrimitive(string id, OrderType op, XVariable *x, int k) {
  if(debug) {
    cout << "\n   constraint  " << id << ":" << x->id << " op " << k << "\n";
  }
  lala::LVar<Allocator> lvar(x->id.c_str());
  constraints.push_back(F::make_binary(F::make_lvar(store_ty, lvar), convert(op), F::make_z(k), store_ty));
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
  throw std::runtime_error("constraint unsupported");
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
  throw std::runtime_error("constraint unsupported");
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
  throw std::runtime_error("constraint unsupported");
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
  throw std::runtime_error("constraint unsupported");
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
  throw std::runtime_error("constraint unsupported");
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
  throw std::runtime_error("constraint unsupported");
}


// string id, vector<XVariable *> &list, int startIndex, XVariable *index, RankType rank, int value
template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildConstraintElement(string, vector<XVariable *> &list, int startIndex, XVariable *index, RankType, int value) {
  if(debug) {
    cout << "\n    element constant (with index) constraint" << endl;
    cout << "        ";
    displayList(list);
    cout << "        value: " << value << endl;
    cout << "        Start index : " << startIndex << endl;
    cout << "        index : " << *index << endl;
  }
  throw std::runtime_error("constraint unsupported");
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
  throw std::runtime_error("constraint unsupported");
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
  throw std::runtime_error("constraint unsupported");
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
  throw std::runtime_error("constraint unsupported");
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
  minimize = lala::LVar<Allocator>(x->id.c_str());
}


template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildObjectiveMaximizeVariable(XVariable *x) {
  if(debug) {
    cout << "\n    objective: maximize variable " << x << endl;
  }
  maximize = lala::LVar<Allocator>(x->id.c_str());
}


template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildObjectiveMinimize(ExpressionObjective type, vector<XVariable *> &list, vector<int> &coefs) {
  if(debug) {
    XCSP3CoreCallbacks::buildObjectiveMinimize(type, list, coefs);
  }
  throw std::runtime_error("constraint unsupported");
}


template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildObjectiveMaximize(ExpressionObjective type, vector<XVariable *> &list, vector<int> &coefs) {
  if(debug) {
    XCSP3CoreCallbacks::buildObjectiveMaximize(type, list, coefs);
  }
  throw std::runtime_error("constraint unsupported");
}


template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildObjectiveMinimize(ExpressionObjective type, vector<XVariable *> &list) {
  if(debug) {
    XCSP3CoreCallbacks::buildObjectiveMinimize(type, list);
  }
  throw std::runtime_error("constraint unsupported");
}


template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildObjectiveMaximize(ExpressionObjective type, vector<XVariable *> &list) {
  if(debug) {
    XCSP3CoreCallbacks::buildObjectiveMaximize(type, list);
  }
  throw std::runtime_error("constraint unsupported");
}

template<class Allocator>
void XCSP3_turbo_callbacks<Allocator>::buildAnnotationDecision(vector<XVariable*> &list) {
  if(debug) {
    std::cout << "       decision variables" << std::endl<< "       ";
    displayList(list);
  }
  throw std::runtime_error("constraint unsupported");
}

#endif // XCSP3_PARSER_HPP
