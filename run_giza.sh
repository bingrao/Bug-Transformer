#!/usr/bin/env bash

# check parameter count and write usage instruction
if (( $# != 1 )); then
  echo "Usage: $0 [small|median|big]"
  exit 1
fi

dataset=$1
#set -ex

############################# Root envs ############################
RootPath=$(pwd)
ProjectPath=${RootPath}/examples/learning_fix
LogPath=${ProjectPath}/logs; [ -d "$LogPath" ] || mkdir -p "$LogPath"
DataPath=${ProjectPath}/data/${dataset}; [ -d "$DataPath" ] || mkdir -p "$DataPath"
MGIZA_DIR="${RootPath}/tools/giza-align/GIZA++-v2"
MKCLS_DIR="${RootPath}/tools/giza-align/mkcls-v2"
SCRIPT_DIR="${RootPath}/tools/giza-align/scripts"



# check if MGIZA_DIR is set and installed
if [ -z "${MGIZA_DIR}" ]; then
  echo "Set the variable MGIZA_DIR"
  exit 1
fi

for mode in "train" "eval" "test"; do
  rm -fr "${DataPath}"/alignment/*

  OutDir=${DataPath}/alignment/${mode}; [ -d "$OutDir" ] || mkdir -p "$OutDir"
  FeedwardDir=${DataPath}/alignment/${mode}/feedward; [ -d "$FeedwardDir" ] || mkdir -p "$FeedwardDir"
  BackwardDir=${DataPath}/alignment/${mode}/backward; [ -d "$BackwardDir" ] || mkdir -p "$BackwardDir"

  source_path="${OutDir}/${mode}-buggy.txt"
  target_path="${OutDir}/${mode}-fixed.txt"
  cp "${DataPath}/${mode}-buggy.txt" "${source_path}"
  cp "${DataPath}/${mode}-fixed.txt" "${target_path}"
  source_name="${mode}-buggy"
  target_name="${mode}-fixed"
  prefix_source="${OutDir}/${source_name}"
  prefix_target="${OutDir}/${target_name}"

  # creates vcb and snt files
  "${MGIZA_DIR}"/plain2snt.out "${source_path}" "${target_path}"

  # Create class for the input
  "${MKCLS_DIR}"/mkcls -m2 -n10 -p"${source_path}" -c50 -V"${prefix_source}.vcb.classes" &
  "${MKCLS_DIR}"/mkcls -m2 -n10 -p"${target_path}" -c50 -V"${prefix_target}.vcb.classes" &
  wait

  # Feedward
  "${MGIZA_DIR}"/snt2cooc.out  "${prefix_source}.vcb" "${prefix_target}.vcb" \
                               "${prefix_source}_${target_name}.snt" > \
                               "${prefix_source}"_${target_name}.cooc &

  wait

  "${MGIZA_DIR}"/GIZA++ -S "${prefix_source}.vcb" \
                        -T "${prefix_target}.vcb" \
                        -C "${prefix_source}_${target_name}.snt" \
                        -CoocurrenceFile "${prefix_source}"_${target_name}.cooc \
                        -outputpath "${FeedwardDir}"

  python "${SCRIPT_DIR}"/a3ToTalp.py < "${FeedwardDir}"/*.AA3.final > "${FeedwardDir}"/${mode}-feedward.talp

  # Backward
  "${MGIZA_DIR}"/snt2cooc.out  "${prefix_target}.vcb" "${prefix_source}.vcb" \
                               "${prefix_target}_${source_name}.snt" > \
                               "${prefix_target}"_${source_name}.cooc &

  wait

  "${MGIZA_DIR}"/GIZA++ -S "${prefix_target}.vcb" \
                        -T "${prefix_source}.vcb" \
                        -C "${prefix_target}_${source_name}.snt" \
                        -CoocurrenceFile "${prefix_target}"_${source_name}.cooc \
                        -outputpath "${BackwardDir}"

  python "${SCRIPT_DIR}"/a3ToTalp.py < "${BackwardDir}"/*.AA3.final > "${BackwardDir}"/${mode}-backward.talp

done