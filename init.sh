#!/bin/bash

singularity shell --nv --bind `readlink $HOME` --bind `readlink -f ${HOME}/nobackup/` --bind /uscms_data/d1/`whoami` --bind /cvmfs /cvmfs/unpacked.cern.ch/registry.hub.docker.com/fnallpc/fnallpc-docker\:tensorflow-latest-gpu-singularity/
