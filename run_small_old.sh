#!/usr/bin/env bash

if [ "$#" -ne 1 ] ; then
  echo "Missing Parameters ..."
  echo "Usage: $0 dataset[small|small_old|median] target[abstract|preprocess|train|translate|all|inference]" >&2
  exit 1
fi

dataset="small_old"
target=$1


############################# Root envs ############################
RootPath=`pwd`/examples/learning_fix
ConfigPath=${RootPath}/config
BinPath=${RootPath}/bin
LogPath=${RootPath}/logs
DataPath=${RootPath}/data


[ -d $ConfigPath ] || mkdir -p $ConfigPath
[ -d $BinPath ] || mkdir -p $BinPath
[ -d $LogPath ] || mkdir -p $LogPath
[ -d $DataPath ] || mkdir -p $DataPath

CurrentDate=$(date +%F)

########################### Project Parameters #######################
# Log file

LogFile=${LogPath}/${dataset}-${target}-${CurrentDate}.log

# Config file for scala application to generate abstract code
ConfigAbstract=${ConfigPath}/application_${dataset}.conf

# Config files for model data preprocess, train, translate
ConfigFile=${ConfigPath}/${dataset}_1.yml

######### Special parameters for model translate ###################

# Test training model checkpoint path for translating
ModelCheckpoint=${DataPath}/${dataset}/${dataset}_step_20000.pt

# The buggy code (source) to translate task
TranslateSource=${DataPath}/${dataset}/test-buggy.txt

# The fixed code (target) to translate task
TranslateTarget=${DataPath}/${dataset}/test-fixed.txt

# The model predict output, each line is corresponding to the line in buggy code
TranslateOutput=${DataPath}/${dataset}/predictions.txt

# The beam size for prediction
TranslateBeamSize=10

#####################################################################
########################### Helper functions  #######################
#####################################################################


function _bleu() {
  echo "------------------- BLEU ------------------------"

  echo "buggy vs fixed"
  ${BinPath}/multi-bleu.perl ${TranslateTarget} < ${TranslateSource}

  echo "prediction vs fixed"
  ${BinPath}/multi-bleu.perl ${TranslateTarget} < ${TranslateOutput}
}

function _classification() {
  echo "------------------- CLASSIFICATION ------------------------"

  total=`wc -l ${TranslateTarget}| awk '{print $1}'`
  echo "Test Set: $total"

  echo "Predictions"
  output=$(python3 ${BinPath}/prediction_classifier.py "${TranslateSource}" "${TranslateTarget}" "${TranslateOutput}" 2>&1)
  perf=`awk '{print $1}' <<< "$output"`
  changed=`awk '{print $2}' <<< "$output"`
  bad=`awk '{print $3}' <<< "$output"`
  perf_perc="$(echo "scale=2; $perf * 100 / $total" | bc)"

  echo "Perf: $perf ($perf_perc%)"
  echo "Pot : $changed"
  echo "Bad : $bad"
}

function _abstract() {
  echo "------------------- Code Abstract ------------------------"
  set -x
  export JAVA_OPTS="-Xmx32G -Xms1g -Xss512M"
  scala ${BinPath}/java_abstract-1.0-jar-with-dependencies.jar ${ConfigAbstract}
}

function _train() {
  echo "------------------- Training ------------------------"
  onmt_train -config ${ConfigFile} -log_file ${LogFile}
}

function _preprocess() {
  echo "------------------- Preprocess  ------------------------"
  onmt_preprocess -config ${ConfigFile} -log_file ${LogFile}
}

function _translate() {
  echo "------------------- Translate  ------------------------"
  beam=$1
  onmt_translate -config ${ConfigFile} \
                 -model ${ModelCheckpoint} \
                 -output ${TranslateOutput} \
                 -tgt ${TranslateTarget} \
                 -src ${TranslateSource} \
                 -log_file ${LogFile} \
                 -beam_size ${beam}
}

function _inference(){
  echo "------------------- Test Beach Search  ------------------------"
  beam_widths=("5" "10" "15" "20" "25" "30" "35" "40" "45" "50" "100" "200")

  for beam_width in ${beam_widths[*]}
  do
    echo "Beam width: $beam_width"
    SECONDS=0
    _translate ${beam_width}
    elapsed=$SECONDS
    echo "---------- Time report ----------"
	  echo "Beam width: $beam_width"
	  echo "Total seconds: $elapsed"

    total=`wc -l ${TranslateSource}| awk '{print $1}'`
    patches=$(($total * $beam_width))
    avg="$(echo "scale=6; $elapsed / $patches" | bc)"
    avg_bug="$(echo "scale=6; $elapsed / $total" | bc)"

    echo "Total bugs: $total"
    echo "Total patches: $patches"
    echo "Avg patch/sec: $avg"
    echo "Avg bug/sec: $avg_bug"
    echo "---------------------------------"

    _bleu

    _classification
  done
}

case ${target} in
   "abstract")
      _abstract
   ;;

   "preprocess")
      _preprocess
   ;;

   "train")
      _train
   ;;

   "translate")
      _translate ${TranslateBeamSize}
      _bleu
      _classification
   ;;

   "all")
      _preprocess
      _train
      _translate ${TranslateBeamSize}
      _bleu
      _classification
   ;;
   "inference")
      _inference
   ;;
   *)
     echo "There is no match case for ${target}"
     echo "Usage: $0 dataset[small|small_old|median] target[abstract|preprocess|train|translate|all|inference]" >&2
     exit 1
   ;;
esac
