#!/bin/bash

### Dedupliate DeepCAD ###
# Deduplicate repeatd CAD B-rep (LDM training)
python deduplicate_cad.py --data deepcad_parsed --bit 6 --option 'deepcad'
# Deduplicate repeated surface & edge (VAE training)
python deduplicate_surfedge.py --data deepcad_parsed --list deepcad_data_split_6bit.pkl --bit 6 --option 'deepcad'
python deduplicate_surfedge.py --data deepcad_parsed --list deepcad_data_split_6bit.pkl --bit 6 --edge --option 'deepcad'


### Dedupliate ABC ###
# Deduplicate repeatd CAD B-rep (LDM training)
python deduplicate_cad.py --data abc_parsed --bit 6 --option 'abc'
# Deduplicate repeated surface & edge (VAE training)
python deduplicate_surfedge.py --data abc_parsed --list abc_data_split_6bit.pkl --bit 6 --option 'abc'
python deduplicate_surfedge.py --data abc_parsed --list abc_data_split_6bit.pkl --bit 6 --edge --option 'abc'


### Dedupliate Furniture ###
# Deduplicate repeatd CAD B-rep (LDM training)
python deduplicate_cad.py --data furniture_parsed --bit 6 --option 'furniture'
# Deduplicate repeated surface & edge (VAE training)
python deduplicate_surfedge.py --data furniture_parsed --list furniture_data_split_6bit.pkl --bit 6 --option 'furniture'
python deduplicate_surfedge.py --data furniture_parsed --list furniture_data_split_6bit.pkl --bit 6 --edge --option 'furniture'