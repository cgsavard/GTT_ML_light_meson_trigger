# Light Meson GNN Clustering for the GTT

This repo is the work product created during the ML@L1 Trigger Workshop held at the Fermi National Accelerator Laboratory's LHC Physics Center on October 10-14, 2022. The project seeks to cluster light mesons (phi and B_s) using inputs from the CMS track trigger.

This project is based off the work of [arXiv:1902.07987](https://arxiv.org/pdf/1902.07987.pdf) ([GitHub](https://github.com/jkiesele/caloGraphNN)), wherein we will make use of the GarNet model. This model has been modified for use with QKeras, making it suitable for implementation on an FPGA. This work is discussed at [arXiv:2008.03601](https://arxiv.org/pdf/2008.03601.pdf) ([GitHub](https://github.com/fastmachinelearning/hls4ml/blob/main/contrib/garnet.py)).

## Setup environment

```bash
source init.sh
./jupy.py <8NNNN>
```

## Data location

20k events of BsToPhiPhi PU200 

```bash
/afs/cern.ch/work/p/ppalit2/public/bstophiphignnsample
```