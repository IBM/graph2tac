#!/bin/sh
mkdir -p dataset
cd _build/default && rsync -avm --include='*.bin' -f 'hide,! */' . ../../dataset
