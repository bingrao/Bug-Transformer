#!/usr/bin/env bash

dataset=$1
configFile=$3
#set -ex

############################# Root envs ############################
RootPath="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ProjectPath=${RootPath}/examples/defect4j
ConfigPath=${ProjectPath}/config/${dataset}; [ -d "$ConfigPath" ] || mkdir -p "$ConfigPath"
MetaData=${ProjectPath}/config/${dataset}/metadata.csv
BinPath=${ProjectPath}/bin; [ -d "$BinPath" ] || mkdir -p "$BinPath"
LogPath=${ProjectPath}/logs; [ -d "$LogPath" ] || mkdir -p "$LogPath"
DataPath=${ProjectPath}/data/${dataset}; [ -d "$DataPath" ] || mkdir -p "$DataPath"
CurrentDate=$(date +%F)

function help() {
     echo "Usage: [export CUDA_VISIBLE_DEVICES=0;] $0 dataset target configFile" >&2
     echo "       dataset: [small|median|big|small_old]"
     echo "       target:  [abstract|preprocess|train|translate|all|inference|performance|loop_translate]"
     echo "Example: Using third (or first by default) GPU to train small dataset with small_1.yml config file"
     echo "Example: The default direcotry that system searches config files: ${ProjectPath}/config/[small|median|big|small_old]"
     echo "       - export CUDA_VISIBLE_DEVICES=2,3; bash run.sh small train small_1.yml"
     echo "       - bash run.sh small train small_1.yml"
}

########################### Project Parameters #######################

prefix="${dataset}-$(echo "${configFile}" | cut -d'.' -f1)"

config_index=$(echo "${configFile}" |  tr -dc '0-9')

# Log file
LogFile=${LogPath}/${CurrentDate}-${prefix}.log

#######################################################################################################
######################################## Helper functions  ############################################
#######################################################################################################

function logInfo() {
    echo "[$(date +"%F %T,%3N") INFO] $1" | tee -a "${LogFile}"
}

## Config files for model data preprocess, train, translate
#ConfigFile=${ConfigPath}/${configFile}
#if [ -f "$ConfigFile" ]; then
#    logInfo "Loading config from $ConfigFile."
#else
#    logInfo "Config file $ConfigFile does not exist."
#    exit 1
#fi

echo "Reading from Defects4J_oneLiner_metadata.csv"
while IFS=, read -r col1 col2 col3 col4
do
  BUG_PROJECT=${DataPath}/${col1}_${col2}
  mkdir -p $BUG_PROJECT
  logInfo "Checking out ${col1}_${col2} to ${BUG_PROJECT}"
  defects4j checkout -p $col1 -v ${col2}b -w $BUG_PROJECT &>/dev/null

#  echo "Generating patches for ${col1}_${col2}"
#  $CURRENT_DIR/../sequencer-predict.sh --buggy_file=$BUG_PROJECT/$col3 --buggy_line=$col4 --beam_size=50 --output=$DEFECTS4J_PATCHES_DIR/${col1}_${col2}
#  echo
#
#  echo "Running test on all patches for ${col1}_${col2}"
#  python3 $CURRENT_DIR/validatePatch.py $DEFECTS4J_PATCHES_DIR/${col1}_${col2} $BUG_PROJECT $BUG_PROJECT/$col3
#  echo
#
#  echo "Deleting ${BUG_PROJECT}"
#  rm -rf $BUG_PROJECT
#  echo
done < ${MetaData}

#echo "Deleting Defects4J_projects"
#rm -rf $DEFECTS4J_DIR
#echo
