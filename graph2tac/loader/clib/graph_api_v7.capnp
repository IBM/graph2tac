@0xb9d31af01976cf9c; # v7

using File = Text;
using DepIndex = UInt16;
using NodeIndex = UInt32;
using TacticId = UInt64;
using DefinitionId = UInt64;

struct Graph {
  # Note: This struct fits exactly in 64 bits. Let's keep it that way.
  struct EdgeTarget {
    label @0 :EdgeClassification;
    target :group {
      depIndex @1 :DepIndex;
      nodeIndex @2 :NodeIndex;
    }
  }
  struct Node { # Fits exactly in 128 bits.
    label :union { # Inlined for efficiency purposes
      root @0 :Void;

      # Context
      contextDef @1 :Text;
      contextAssum @2 :Text;

      # Definitions
      definition @3 :Definition;
      constEmpty @4 :Void;

      # Sorts
      sortSProp @5 :Void;
      sortProp @6 :Void;
      sortSet @7 :Void;
      sortType @8 :Void; # Collapsed universe

      # Constr nodes
      rel @9 :Void;
      var @10 :Void;
      evar @11 :IntP; #TODO: Resolve
      evarSubst @12 :Void;
      cast @13 :Void;
      prod @14 :Void;
      lambda @15 :Void;
      letIn @16 :Void;
      app @17 :Void;
      appFun @18 :Void;
      appArg @19 :Void;
      case @20 :Void;
      caseBranch @21 :Void;
      fix @22 :Void;
      fixFun @23 :Void;
      coFix @24 :Void;
      coFixFun @25 :Void;

      # Primitives
      int @26 :IntP;
      float @27 :FloatP;
      primitive @28 :Text;
    }

    childrenIndex @29 :UInt32;
    childrenCount @30 :UInt16;
  }
  # The main memory store of the graph. It acts as a heap similar to the main memory of a C/C++ program.
  # The heap is accessed by indexing the `nodes` list using a `NodeIndex` which returns a `Node`.
  # Every node has a label and a list of children, which is indicated as a range within the `edges` list using
  # `childrenIndex` and `childrenCount`. The targets of the edges can again be found in the `nodes` list of the
  # current file or of a dependency.
  # Note that just like in C/C++ doing pointer arithmetic on the heap is undefined behavior, and you may
  # encounter arbitrary garbage if you do this. In particular, iterating over the heap is discouraged.
  nodes @0 :List(Node);
  edges @1 :List(EdgeTarget);
}

struct ProofState {
  root @0 :NodeIndex;
  context @1 :List(NodeIndex);
  text @2 :Text;
}

struct AbstractTactic {
  ident @0 :TacticId;
  parameters @1 :UInt8;
}

struct Tactic {
  # Together with the tag, this fits exactly in 64 bits. Lets keep it that way.
  struct Argument {
    union {
      unresolvable @0 :Void;
      term :group {
        depIndex @1 :DepIndex;
        nodeIndex @2 :NodeIndex;
      }
    }
  }

  ident @0 :TacticId;
  arguments @1 :List(Argument);
  text @2 :Text; # WARNING: This is currently not 1-to-1 isomorphic to (ident, arguments)!
  # A textual representation of the base tactic without arguments. It tries to roughly correspond to `ident`.
  # Note, however, that this is a slight under-approximation, because tactic printing is not 100% isomorphic to
  # Coq's internal AST of tactics. As such, there are slightly more unique `ident`'s than `bareText`'s in the dataset.
  baseText @3 :Text;
  intermText @4 :Text;
}

struct Dataset {
  # The first file is always the current file
  dependencies @0 :List(File);
  graph @1 :Graph;
  tacticalDefinitions @2 :List(NodeIndex);
}

struct Exception {
  union {
    noSuchTactic @0 :Void;
    mismatchedArguments @1 :Void;
    parseError @2 :Void;
    illegalArgument @3 :Void;
  }
}

struct ExecutionResult {
  union {
    failure @0 :Void;
    complete @1 :Void;
    newState :group {
      graph @2 :Graph;
      state @3 :ProofState;
      obj @4 :ProofObject;
    }
    protocolError @5 :Exception;
  }
}

interface ProofObject {
  runTactic @0 (tactic: Tactic) -> (result: ExecutionResult);
}

interface AvailableTactics {
  tactics @0 () -> (tactics :List(AbstractTactic));
  printTactic @1 (tactic :TacticId) -> (tactic :Text);
}

interface PullReinforce {
  reinforce @0 (lemma :Text) -> (available :AvailableTactics, result :ExecutionResult);
}

interface PushReinforce {
  reinforce @0 (result :ExecutionResult);
  embed @1 (graph :Graph, root :NodeIndex) -> (emb :List(Float64));
}

interface Main {
  initialize @0 (push :PushReinforce) -> (pull :PullReinforce);
}

struct ProofStep {
  state @0 :ProofState;
  tactic @1 :Tactic;
}

struct Definition {
  hash @0 :DefinitionId;
  name @1 :Text;

  union {
    inductive @2 :Void;
    constructor @3 :Void;
    projection @4 :Void;

    # A constant defined by directly inputting a term
    # In the future, we might augment such constants with tactical
    # refinement proofs that build the term iteratively.
    manualConstant @5 :Void;

    # A constant that was either directly or indirectly using a tactical proof.
    tacticalConstant :group {

      # The tactical proof associated to the constant.
      tacticalProof @6 :List(ProofStep);
    }
  }
}

# Used for in-mermory space optimization. This allows us to make structs smaller by reusing space in
# the pointer section of a struct that would otherwise be allocated in the data section.
struct FloatP {
  value @0 :Float64;
}
struct IntP {
  value @0 :UInt64;
}


enum EdgeClassification {
  # Contexts
  contextElem @0;
  contextSubject @1;

  # Context elements
  contextDefType @2;
  contextDefTerm @3;

  # Constants
  constType @4;
  constUndef @5;
  constDef @6;
  constOpaqueDef @7;
  constPrimitive @8;

  # Inductives
  indType @9;
  indConstruct @10;
  projTerm @11;
  constructTerm @12;

  # Casts
  castTerm @13;
  castType @14;

  # Products
  prodType @15;
  prodTerm @16;

  # Lambdas
  lambdaType @17;
  lambdaTerm @18;

  # LetIns
  letInDef @19;
  letInType @20;
  letInTerm @21;

  # Apps
  appFunPointer @22;
  appFunValue @23;
  appArgPointer @24;
  appArgValue @25;
  appArgOrder @26;

  # Cases
  caseTerm @27;
  caseReturn @28;
  caseBranchPointer @29;
  caseInd @30;

  # CaseBranches
  cBConstruct @31;
  cBTerm @32;

  # Fixes
  fixMutual @33;
  fixReturn @34;

  # FixFuns
  fixFunType @35;
  fixFunTerm @36;

  # CoFixes
  coFixMutual @37;
  coFixReturn @38;

  # CoFixFuns
  coFixFunType @39;
  coFixFunTerm @40;

  # Constr edges
  relPointer @41;
  varPointer @42;
  evarSubstPointer @43;
  evarSubstOrder @44;
  evarSubstValue @45;
}

# Struct is needed to work around
# https://github.com/capnproto/capnp-ocaml/issues/81
struct ConflatableEdges {
  conflatable @0 :List(EdgeClassification);
}
const conflatableEdges :List(ConflatableEdges) =
[ ( conflatable = [contextDefType, constType, indType, castType, prodType, lambdaType, letInType, fixFunType, coFixFunType] )
, ( conflatable = [contextDefTerm, castTerm, prodTerm, lambdaTerm, letInTerm, fixFunTerm, coFixFunTerm] )
# Not conflatable: projTerm, constructTerm, caseTerm, cBTerm
, ( conflatable = [varPointer, relPointer] )
, ( conflatable = [appArgOrder, evarSubstOrder] )
];
const importantEdges :List(EdgeClassification) =
[ contextElem, contextSubject, contextDefType, contextDefTerm, constType, constDef, constOpaqueDef, indType, indConstruct, constructTerm
, prodType, prodTerm, lambdaType, lambdaTerm, letInDef, letInType, letInTerm, appFunPointer, appArgPointer, appArgOrder, relPointer, varPointer ];
const lessImportantEdges :List(EdgeClassification) =
[ caseTerm, caseReturn, caseBranchPointer, caseInd, cBConstruct, cBTerm, fixMutual, fixReturn, fixFunType, fixFunTerm ];
const leastImportantEdges :List(EdgeClassification) =
[ constUndef, constPrimitive, projTerm, castTerm, castType, appFunValue, appArgValue, coFixReturn, coFixFunType, coFixFunTerm
, coFixMutual, evarSubstPointer, evarSubstOrder, evarSubstValue ];

# WARNING: DO NOT USE
# This is just for visualization purposes in order to drastically reduce the number of edges. You should not use it in networks
const groupedEdges :List(ConflatableEdges) =
[ ( conflatable = [contextElem, contextSubject] )
, ( conflatable = [contextDefType, contextDefTerm] )
, ( conflatable = [constType, constUndef, constDef, constOpaqueDef, constPrimitive] )
, ( conflatable = [indType, indConstruct] )
, ( conflatable = [projTerm] )
, ( conflatable = [constructTerm] )
, ( conflatable = [castType, castTerm] )
, ( conflatable = [prodType, prodTerm] )
, ( conflatable = [lambdaType, lambdaTerm] )
, ( conflatable = [letInDef, letInTerm, letInType] )
, ( conflatable = [appFunPointer, appArgPointer, appArgOrder] )
, ( conflatable = [appFunValue] )
, ( conflatable = [appArgValue] )
, ( conflatable = [caseTerm, caseReturn, caseBranchPointer, caseInd] )
, ( conflatable = [cBConstruct, cBTerm] )
, ( conflatable = [fixMutual, fixReturn] )
, ( conflatable = [fixFunType, fixFunTerm] )
, ( conflatable = [coFixMutual, coFixReturn] )
, ( conflatable = [coFixFunType, coFixFunTerm] )
, ( conflatable = [relPointer] )
, ( conflatable = [varPointer] )
, ( conflatable = [evarSubstPointer] )
, ( conflatable = [evarSubstOrder, evarSubstValue] )
];
