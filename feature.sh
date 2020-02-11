#!/bin/bash

TARGET="T1019s2"
TARGET_DIR="test_data"
TARGET_SEQ="${TARGET_DIR}/${TARGET}.seq" # fasta format
PLMDCA_DIR="plmDCA/plmDCA_asymmetric_v2/"

# generate domain crops from target seq
python feature.py -s $TARGET_SEQ -c

for domain in ${TARGET_DIR}/*.seq; do
	out=${domain%.seq}
	echo "Generate MSA files for ${out}"
	hhblits -cpu 4 -i ${out}.seq -d databases/uniclust30_2018_08/uniclust30_2018_08 -oa3m ${out}.a3m -ohhm ${out}.hhm -n 3
	reformat.pl ${out}.a3m ${out}.fas
	psiblast -subject ${out}.seq -in_msa ${out}.fas -out_ascii_pssm ${out}.pssm
done

# make target features data and generate ungap target aln file for plmDCA
python feature.py -s $TARGET_SEQ -f

cd $PLMDCA_DIR
for aln in ../../${TARGET_DIR}/*.aln; do
	echo "calculate plmDCA for $aln"
	octave plmDCA.m $aln
done
cd -

# run again to update target features data
python feature.py -s $TARGET_SEQ -f