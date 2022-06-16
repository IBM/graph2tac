import os
from pathlib import Path



def get_all_fnames(graphs_dir: Path, pattern: str):
    yield from sorted(str(os.fspath(f.relative_to(graphs_dir).with_suffix('')))
                for f in Path(graphs_dir).glob(pattern) if f.is_file())



class DataFile:
    def __init__(self, api, graphs_dir: Path, fname):
        self.fname = fname
        self.graphs_dir = graphs_dir
        self.api = api

    def __enter__(self):
        self.f = open(os.path.join(self.graphs_dir, "{}.bin".format(self.fname)))
        self.data = self.api.Dataset.read_packed(self.f, traversal_limit_in_words=2**64-1)
        return self.data

    def __exit__(self, *exception_data):
        self.data.graph.edges  # TODO(jrute): See slack for a more general way to do this
        self.f.close()

def collect_proof_steps(dataset):
    # print(f"found {len(dataset.tacticalDefinitions)} tacticial definitions")
    for lemma_i in dataset.tacticalDefinitions:
        node = dataset.graph.classifications[lemma_i].definition
    #for node in dataset.graph.classifications:
    #    if node.which != "definition": continue
    #    node = node.definition
    #    if node.which != "tacticalConstant": continue

        lemma_hash_name = (node.hash, node.name)
        steps = node.tacticalConstant.tacticalProof
        for step in steps:
            state = step.state
            tactic = step.tactic
            root = step.state.root
            context = list(step.state.context)
            state_text = step.state.text
            tactic_h = step.tactic.ident
            tactic_args = list(step.tactic.arguments)
            tactic_text = step.tactic.text
            yield (
                lemma_hash_name,
                (state.root, list(state.context), state.text),
                (tactic.ident, list(tactic.arguments), tactic.text)
            )
