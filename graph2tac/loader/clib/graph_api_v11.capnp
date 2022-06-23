@0xf70bdd1a2cb44721; # v11

######################################################################################################
#
#               Tactician Graph Encoding Schema File
#
# This file currently serves three different purposes, each with a different entry-point:
#
# 1. A schema for graph-based dataset exported from Coq. The entry-point for this is the `Dataset`
#    struct. That is, every file in the dataset contains a message conforming to this struct.
#
# 2. A communication protocol for reinforcement learning. Depending on who initiates the learning
#    (Coq or the learning agent), the entry-point for this is respectively the `PushReinforce` or
#    the `PullReinforce` interface.
#
# 3. A communication protocol for proof synthesis with Tactician. The entry-point of this
#    protocol is the `PredictionProtocol` struct.
#
# All these three entry-points share a common base, which encodes the notion of graphs, terms,
# tactics and more.
#
######################################################################################################


######################################################################################################
#
#             Type aliases
#
######################################################################################################

using File = Text;
# The URI of a file in a dataset, relativized w.r.t. the root of the dataset

using NodeIndex = UInt32;
# A `NodeIndex` is the identity of a node in a graph
using DepIndex = UInt16;
# A `DepIndex` comes together with a `NodeIndex` and indicates to which graph a node belongs.
# How a dependency index should be resolved to a graph is not specified here. However, a
# dependency index of zero always refers to the 'current' graph.

using TacticId = UInt64;
using DefinitionId = UInt64;
using ProofStateId = UInt32; # Note, proof state ids are only unique within their own proof
# Tactics, definitions and proof states are identified by hashes


######################################################################################################
#
#             Common datastructures
#
######################################################################################################

struct Graph {
  # A graph is the main data store (the 'heap') that contains the bulk of the data in the dataset and
  # during communication with Coq. A graph is a collection of labeled nodes with directed, labeled edges
  # between them. A graph may reference nodes from other graphs. For an edge 'A --> B' in the graph it
  # always holds that 'A' is part of the current graph, but 'B' may (or may not) be part of another graph.

  struct EdgeTarget { # Fits exactly in 64 bits. Let's keep it that way.
    # The 'pointy' end of an edge, together with the label of that edge.

    label @0 :EdgeClassification;
    target :group {
      depIndex @1 :DepIndex;
      # Indicates to which graph a node belongs. How this should be resolved is not specified here.
      # However, a value of zero always points to the 'current' graph.

      nodeIndex @2 :NodeIndex;
      # The index into `Graph.nodes` where this node can be found.
    }
  }

  struct Node { # Fits exactly in 128 bits.
    # A node has a label that optionally contains some additional information, together with a list of
    # outgoing edges (children).

    label :union { # Inlined for efficiency purposes
      proofState @0 :Void;
      undefProofState @28 :Void;

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
      evar @10 :Void;
      evarSubst @11 :Void;
      cast @12 :Void;
      prod @13 :Void;
      lambda @14 :Void;
      letIn @15 :Void;
      app @16 :Void;
      appFun @17 :Void;
      appArg @18 :Void;
      case @19 :Void;
      caseBranch @20 :Void;
      fix @21 :Void;
      fixFun @22 :Void;
      coFix @23 :Void;
      coFixFun @24 :Void;

      # Primitives
      int @25 :IntP;
      float @26 :FloatP;
      primitive @27 :Text;
    }

    childrenIndex @29 :UInt32;
    childrenCount @30 :UInt16;
    # The children of a node are encoded as a range withing the `edge`-list of the graph.
  }

  nodes @0 :List(Node);
  edges @1 :List(EdgeTarget);
  # The main memory store of the graph. It acts as a heap similar to the main memory of a C/C++ program.
  # The heap is accessed by indexing the `nodes` list using a `NodeIndex` which returns a `Node`.
  # Every node has a label and a list of children, which is indicated as a range within the `edges` list using
  # `childrenIndex` and `childrenCount`. The targets of the edges can again be found in the `nodes` list of the
  # current file or of a dependency.
  # Note that just like in C/C++ doing pointer arithmetic on the heap is undefined behavior, and you may
  # encounter arbitrary garbage if you do this. In particular, iterating over the heap is discouraged.
  # Instead, you should access the heap through various entry-points that are provided.
}

struct AbstractTactic {
  ident @0 :TacticId;
  # An abstract tactic is referenced to using a identifier (hash).

  parameters @1 :UInt8;
  # Every tactic has a constant number of parameters that need to be filled in.
}

struct Tactic {
  # A concrete tactic with it's parameters determined. Somewhat strangely, this struct does not actually include
  # these parameters. They can instead be found in `Outcome.tacticArguments`. The reason for this is that one
  # tactic can run on multiple proof states at the same time and for all of those proof states, the arguments
  # may be resolved differently.

  ident @0 :TacticId;

  text @1 :Text;
  # The full text of the tactic including the full arguments. This does not currently correspond to
  # (ident, arguments) because in this dataset arguments do not include full terms, but only references to
  # definitions and local context elements.

  baseText @2 :Text;
  # A textual representation of the base tactic without arguments. It tries to roughly correspond to `ident`.
  # Note, however, that this is both an under-approximation and an over-approximation. The reason is that tactic
  # printing is not 100% isomorphic to Coq's internal AST of tactics. Sometimes, different tactics get mapped to
  # the same text. Conversely, the same tactic may be mapped to different texts when identifiers are printed in
  # using different partially-qualified names.

  intermText @3 :Text;
  # A textual representation that tries to come as close as possible to (ident, arguments).
  # It comes with the same caveats as `baseText`.

  exact @4 :Bool;
  # Indicates whether or not `ident` + `arguments` is faithfully reversible into the original "strictified" tactic.
  # Note that this does not necessarily mean that it represents exactly the tactic that was inputted by the user.
  # All tactics are modified to be 'strict' (meaning that tactics that have delayed variables in them break).
  # This flag measures the faithfulness of the representation w.r.t. the strict version of the tactic, not the
  # original tactic inputted by the user.
}

struct Argument {
  # A concrete argument of a tactic.
  union {
    unresolvable @0 :Void;
    # An argument that is currently unresolvable due to limitations of the extraction process.

    term :group {
      # The root of a graph representing an argument that is a term in the calculus of constructions.
      depIndex @1 :DepIndex;
      nodeIndex @2 :NodeIndex;
    }
  }
}

struct ProofState {
  # A proof state represents a particular point in the tactical proof of a constant.

  root @0 :NodeIndex;
  # The entry-point of the proof state, all nodes that are 'part of' the proof state are reachable from here.

  context @1 :List(NodeIndex);
  # The local context of the proof state. These nodes are either `contextAssum` or `contextDef`. Note that
  # these nodes are also reachable from the root of the proof state.

  text @2 :Text;
  # A textual representation of the proof state. This field is only populated in the dataset, not while interacting
  # with Coq.

  id @3 :ProofStateId;
  # A unique identifier of the proof state. Any two proof states in a tactical proof that have an equal id
  # can morally be regarded to be 'the same' proof state.
  # IMPORTANT: Two proof states with the same id may still have different contents. This is because proof states
  #            can contain existential variables (represented by the `evar` node) that can be filled as a
  #            side-effect by a tactic running on another proof state.
}

struct Outcome {
  # An outcome is the result of running a tactic on a proof state. A tactic may run on multiple proof states.

  before @0 :ProofState;
  after @1 :List(ProofState);

  term @2 :NodeIndex;
  # The proof term that witnesses the transition from the before state to the after states. It contains a hole
  # (an `evar` node) for each of the after states. It may also refer to elements of the local context of the
  # before state.

  termText @3 :Text;
  # A textual representation of the proof term.

  tacticArguments @4 :List(Argument);
  # The arguments of the tactic that produced this outcome. Note that these arguments belong to the tactic in
  # `ProofStep.tactic`.
}

struct ProofStep {
  # A proof step is the execution of a single tactic on one or more proof states, producing a list of outcomes.

  tactic :union {
    unknown @0 :Void;
    # Sometimes a tactic cannot or should not be recorded. In those cases, it is marked as 'unknown'.
    # This currently happens with tactics that are run as a result of the `Proof with tac` construct and it
    # happens for tactics that are known to be unsafe like `change_no_check`, `fix`, `cofix` and more.
    known @1 :Tactic;
  }
  outcomes @2 :List(Outcome);
}

struct Definition {
  hash @0 :DefinitionId;
  # A hash of the definition. It is currently not guaranteed to be unique, its usage is currently discouraged.

  name @1 :Text;
  # The name of the definition. The name should be unique in a particular super-global context, but is not unique
  # among different branches of a global context.

  previous @2 :NodeIndex;
  # The previous definition within the global context of the current file.
  # For the first definition this field is set to `len(graph.nodes)`.
  # Attempts are made to make the ordering of the global context consistent with the ordering of definitions
  # in the source document. However, when closing modules and sections this ordering is not guaranteed to be
  # maintained.
  # The contract on this field is that any definition nodes reachable from the forward closure of the definition
  # must also be reachable through the chain of previous fields. An exception to this rule are mutually
  # recursive definitions. Those nodes are placed into the global context in an arbitrary ordering.

  externalPrevious @3 :List(DepIndex);
  # This field provides information about the external global context.
  # At any point in a source file other files 'X' can be loaded through 'Require X'. When this happens, the
  # definitions of X that are reachable through its 'representative' field become available to all subsequent
  # definitions.

  status :union {
    # A definition is either
    # (1) an object as originally inputted by the user
    # (2) a definition that was originally defined in a section and has now had the section
    #     variables discharged into it.
    # (3) a definition that was obtained by performing some sort of module functor substitution.
    # When a definition is not original, we cross-reference to the definition that it was derived from.
    original @4 :Void;
    discharged @5 :NodeIndex;
    substituted :group {
      depIndex @6 :DepIndex;
      nodeIndex @7 :NodeIndex;
    }
  }

  union {
    inductive @8 :Void;
    constructor @9 :Void;
    projection @10 :Void;

    manualConstant @11 :Void;
    # A constant defined by directly inputting a term
    # In the future, we might augment such constants with tactical
    # refinement proofs that build the term iteratively.

    tacticalConstant @12 :List(ProofStep);
    # A constant that was either directly or indirectly generated by a tactical proof.
    # Note that for non-original constants, the proof step sequence may not be very accurate.

    manualSectionConstant @13 :Void;
    # A section variable or local section definition.

    tacticalSectionConstant @14 :List(ProofStep);
    # Local section definitions can also be defined using a tactical proof.
  }

  typeText @15 :Text;
  termText @16 :Text;
  # A textual representation of the type and term of the definition.
  # For inductives, constructors, projections, section variables and axioms the `termText` string
  # is empty.
}


######################################################################################################
#
#             The schema for the dataset
#
######################################################################################################

struct Dataset {
  # Every file in the dataset contains a single message of type `Dataset`. Every file corresponds to
  # a Coq source file, and contains a representation of all definitions that have existed during
  # throughout the compilation of the source file.

  dependencies @0 :List(File);
  # The graph contained in a file may reference nodes from the graph of other files. This field maps
  # a `DepIndex` into the a file that contains that particular node.
  # The first file in this list is always the current file.

  graph @1 :Graph;

  representative @2 :NodeIndex;
  # The entry point of the global context of definitions that are available when this file is 'Required' by
  # another file. The full global context can be obtained by following the `previous` node of definitions.
  # If the compilation unit does not contain any 'super'-global definitions this is set to `len(graph.nodes)`

  definitions @3 :List(NodeIndex);
  # All of the definitions present in the graph.
  # Note that some of these nodes may not be part of the 'super-global' context that is reachable using the
  # `representative` field as an entry point. The reason is that the global context is a forest (list of tree's)
  # and the 'super-global' context is only the main spine of this forest.
}


######################################################################################################
#
#             The schema for reinforcement learning
#
######################################################################################################

struct Exception {
  # A list of things that can go wrong during reinforcement learning.
  union {
    noSuchTactic @0 :Void;
    mismatchedArguments @1 :Void;
    parseError @2 :Void;
    illegalArgument @3 :Void;
  }
}

struct ExecutionResult {
  # The result of executing a tactic on a proof state.

  union {
    failure @0 :Void;
    # The tactic execution failed. This is not an error condition, but rather the natural failure of the tactic.

    complete @1 :Void;
    # The proof has been completed.

    newState :group {
      # The tactic ran successfully and produced a new proof state.
      graph @2 :Graph;
      state @3 :ProofState;
      obj @4 :ProofObject;
      # The proof object can be used to run subsequent tactics on the new proof state.
    }

    protocolError @5 :Exception;
    # Indicates a programmer error.
  }
}

interface ProofObject {
  # Represents a particular proof state.
  runTactic @0 (tactic: Tactic, arguments: List(Argument)) -> (result: ExecutionResult);
  # Run a tactic on the proof state. This function can be called repeatedly, and the given tactic will always be
  # executed on the same proof state.
}

interface AvailableTactics {
  # A way of receiving information about available tactics in a reinforcement learning session.
  tactics @0 () -> (tactics :List(AbstractTactic));
  printTactic @1 (tactic :TacticId) -> (tactic :Text);
}

interface PullReinforce {
  reinforce @0 (lemma :Text) -> (available :AvailableTactics, result :ExecutionResult);
  # An interface allowing a reinforcement learning session to be initiated by the agent.
  # The `lemma` argument is the statement the agent wants to prove. As a response, Coq sends the available
  # tactics that can be used during the proof and the execution result that represents the opening of the
  # session.
}

interface PushReinforce {
  reinforce @0 (result :ExecutionResult);
  # An interface allowing a reinforcement learning session to be initiated by Coq. In this case, Coq decides
  # what lemma should be proved and immediately presents the agent with the initial execution result.
}


######################################################################################################
#
#             The schema for synthesizing proofs using the 'synth' tactic
#
######################################################################################################

struct PredictionProtocol {
  # This protocol works by exchanging raw messages over a socket. The protocol is fully linear.
  # Coq sends a `Request` and waits for a corresponding `Response` message. A message of a given
  # type should be responded with using the obviously corresponding response type.

  struct Request {
    union {
      initialize :group {
        # Start a context for making tactical predictions for proof search. The context includes the tactics
        # that are currently available, the definitions that are available.
        tactics @0 :List(AbstractTactic);
        graph @1 :Graph;
        definitions @2 :List(NodeIndex);
        logAnnotation @3 :Text;
      }
      predict :group {
        # Request a list of tactic predictions given the graph of a proof state.
        graph @4 :Graph;
        # The graph may reference definitions that were transmitted during the the `initialize` message.
        # This graph is designated using the `DepIndex` 1.
        state @5 :ProofState;
      }
      synchronize @6 :UInt64;
      # Coq uses this message to synchronize the state of the protocol when exceptions have occurred.
      # The contract is that the given integer needs to be echo'd back verbatim.
      checkAlignment :group {
        # Request for the server to align the given tactics and definition to it's internal knowledge
        # and report back any tactics and definitions that were not found
        tactics @7 :List(AbstractTactic);
        graph @8 :Graph;
        definitions @9 :List(NodeIndex);
      }
    }
  }
  struct Prediction {
    tactic @0 :Tactic;
    arguments @1 :List(Argument);
    confidence @2 :Float64;
  }
  struct TextPrediction {
    tacticText @0 :Text;
    confidence @1 :Float64;
  }
  struct Response {
    # See Request for documentation.
    union {
      initialized @0 :Void;
      prediction @1 :List(Prediction);
      textPrediction @2 :List(TextPrediction);
      # Output is a list of predictions with a confidence. The list is expected to be
      # sorted by decreasing confidence.
      synchronized @3 :UInt64;
      alignment :group {
        unalignedTactics @4 :List(TacticId);
        unalignedDefinitions @5 :List(NodeIndex);
      }
    }
  }
}


######################################################################################################
#
#             Common datastructures that are too long, cumbersome, and uninteresting
#             to put at the beginning of this file.
#
######################################################################################################

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

  # Back pointers
  relPointer @41;

  # Evars
  evarSubstPointer @42;
  evarSubstTerm @43;
  evarSubstTarget @44;
  evarSubject @45;
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
];
const importantEdges :List(EdgeClassification) =
[ contextElem, contextSubject, contextDefType, contextDefTerm, constType, constDef, constOpaqueDef, indType, indConstruct, constructTerm
, prodType, prodTerm, lambdaType, lambdaTerm, letInDef, letInType, letInTerm, appFunPointer, appArgPointer, appArgOrder, relPointer ];
const lessImportantEdges :List(EdgeClassification) =
[ caseTerm, caseReturn, caseBranchPointer, caseInd, cBConstruct, cBTerm, fixMutual, fixReturn, fixFunType, fixFunTerm ];
const leastImportantEdges :List(EdgeClassification) =
[ constUndef, constPrimitive, projTerm, castTerm, castType, appFunValue, appArgValue, coFixReturn, coFixFunType, coFixFunTerm
, coFixMutual, evarSubstPointer, evarSubstTerm, evarSubstTarget ];

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
, ( conflatable = [evarSubject, evarSubstPointer] )
, ( conflatable = [evarSubstTerm, evarSubstTarget] )
];
