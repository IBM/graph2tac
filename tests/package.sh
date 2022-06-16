#!/bin/sh
mkdir -p dataset
cd _build/default && rsync -avm --include='*.bin' -f 'hide,! */' . ../../dataset
cp $(opam var coq-tactician-reinforce:share)/graph_api.capnp ../../dataset/
