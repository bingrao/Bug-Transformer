#!/bin/bash

set -ex

MGIZA_DIR="/home/bing/project/Bug-Transformer/tools/giza-align/GIZA++-v2"
MKCLS_DIR="/home/bing/project/Bug-Transformer/tools/giza-align/mkcls-v2"

SCRIPT_DIR=${0%/giza.sh}

# check if MGIZA_DIR is set and installed
if [ -z ${MGIZA_DIR} ]; then
  echo "Set the variable MGIZA_DIR"
  exit 1
fi

#if [ ! -f ${MGIZA_DIR}/mgizapp/bin/mgiza ]; then
#  echo "Install mgiza, file ${MGIZA_DIR}/mgizapp/bin/mgiza not found"
#  exit 1
#fi

# check parameter count and write usage instruction
if (( $# != 3 )); then
  echo "Usage: $0 source_file_path target_file_path ln_pair"
  exit 1
fi

source_path=`realpath $1`
target_path=`realpath $2`
source_name=${1##*/}
target_name=${2##*/}
ln_pair=${3}

mkdir -p ${ln_pair}
cd ${ln_pair}

# creates vcb and snt files
${MGIZA_DIR}/plain2snt.out ${source_path} ${target_path}

${MKCLS_DIR}/mkcls -n10 -p${source_path} -V${source_name}.class &
${MKCLS_DIR}/mkcls -n10 -p${target_path} -V${target_name}.class &
wait

${MGIZA_DIR}/snt2cooc.out ${source_name}_${target_name}.cooc ${source_path}.vcb ${target_path}.vcb ${source_path}_${target_name}.snt &
${MGIZA_DIR}/snt2cooc.out ${target_name}_${source_name}.cooc ${target_path}.vcb ${target_path}.vcb ${target_path}_${source_name}.snt &
wait


mkdir -p Forward && cd $_
echo "C ${source_path}_${target_name}.snt" > config.txt
echo "S ${source_path}.vcb" >> config.txt
echo "T ${target_path}.vcb" >> config.txt
echo "CoocurrenceFile ../${source_name}_${target_name}.cooc" >> config.txt
#echo "sourcevocabularyclasses ../${source_name}.class" >> config.txt
#echo "targetvocabularyclasses ../${target_name}.class" >> config.txt

cd ..

mkdir -p Backward && cd $_
echo "C ${target_path}_${source_name}.snt" > config.txt
echo "S ${target_path}.vcb" >> config.txt
echo "T ${source_path}.vcb" >> config.txt
echo "CoocurrenceFile ../${target_name}_${source_name}.cooc" >> config.txt
#echo "sourcevocabularyclasses ../${target_name}.class" >> config.txt
#echo "targetvocabularyclasses ../${source_name}.class" >> config.txt
cd ..

for name in "Forward" "Backward"; do
  cd $name
    # make sure to dump everything [onlineMgiza++](https://307d7cc8-a-db0463cf-s-sites.googlegroups.com/a/fbk.eu/mt4cat/file-cabinet/onlinemgiza-1.0.5-manual.pdf) neeeds
    echo "nodumps 0" >> config.txt
    echo "onlyaldumps 1" >> config.txt
    echo "hmmdumpfrequency 5" >> config.txt
    # Run Giza
    ${MGIZA_DIR}/GIZA++ config.txt
    cat *A3.final.part* > allA3.txt
  cd ..
done

cd ..

# convert alignments
${SCRIPT_DIR}/a3ToTalp.py < ${ln_pair}/Forward/allA3.txt > ${ln_pair}.talp
${SCRIPT_DIR}/a3ToTalp.py < ${ln_pair}/Backward/allA3.txt > ${ln_pair}.reverse.talp

