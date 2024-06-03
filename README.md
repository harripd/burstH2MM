# burstH2MM

[![Tests](https://github.com/harripd/burstH2MM/actions/workflows/Test.yml/badge.svg)](https://github.com/harripd/burstH2MM/actions)

[![Documentation Status](https://readthedocs.org/projects/bursth2mm/badge/?version=latest)](https://bursth2mm.readthedocs.io/en/latest/?badge=latest)

## Introduction

burstH2MM is a package designed to make processing FRETBursts data with photon-by-photon hidden Markov Modeling (H<sup>2</sup>MM) easy to use, and to calculate things such as E, S, transition rates, and other dwell and model parameters, removing the need to manually recode such details each time.

The basic units of burstH2MM are:

- **BurstData** the container of a set of photon streams
- **H2MM_list** The container for a divisor scheme
- **H2MM_resutl** A H<sup>2</sup>MM optimization. This is contains most analysis parameters. Things such as dwell E, dwell S, and dwell mean nanotime

While each class can be assigned to a new variable or list, all three classes keep a record of the object they create, and the object that created them. Therefore it is generally encouraged to simply work with the originating BurstData object and refer to all subfield through it following the appropirate referencing of attributes.

# Documentation

See full documentation at [https://bursth2mm.readthedocs.io](https://bursth2mm.readthedocs.io)

# Installation

This package can be installed with

```bash
$ pip install burstH2MM
```

# Background

H<sup>2</sup>MM was developed by [Pirchi and Tsukanov et. al. J. Phys. Chem. B 2016](https://doi.org/10.1021/acs.jpcb.6b10726), and extended in [Harris et.al. Nat. Comm. 2022](https://doi.org/10.1038/s41467-022-28632-x). This modeule requies the H2MM_C packaged introduced in *Harris et.al.*, and [FRETBursts](https://fretbursts.readthedocs.io/en/latest/), developed by [Ingariola et. al. Plos ONE 2016](https://doi.org/10.1371/journal.pone.0160716) to make the process of anlaysis far easier for single molecule FRET data.

When using `burstH2MM` please cite all three papers.