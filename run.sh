#!/bin/bash

if [ "$#" -ne 1 ] ; then
  echo "Missing Parameters ..."
  echo "Usage: $0 target[abstract|preprocess|train|translate|all]" >&2
  exit 1
fi


target=$1

############################# Root envs ############################
RootPath=`pwd`/examples/learning_fix
ConfigPath=${RootPath}/config
BinPath=${RootPath}/bin
CurrentDate=$(date +%F)

############################
LogFile=${RootPath}/logs/${target}-${CurrentDate}.log
ConfigAbstract=${ConfigPath}/application_small.conf
ConfigFile=${ConfigPath}/small_1.yml
ModelCheckpoint=${RootPath}/data/small/small_step_20000.pt
TranslateSource=${RootPath}/data/small/test-buggy.txt
TranslateTarget=${RootPath}/data/small/test-fixed.txt
TranslateOutput=${RootPath}/data/small/predictions.txt
TranslateBeamSize=10
_bleu() {
  echo "------------------- BLEU ------------------------"

  echo "buggy vs fixed"
  ${BinPath}/multi-bleu.perl ${TranslateTarget} < ${TranslateSource}

  echo "prediction vs fixed"
  ${BinPath}/multi-bleu.perl ${TranslateTarget} < ${TranslateOutput}
}

_classification() {
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



_abstract() {
  set -x
  export JAVA_OPTS="-Xmx32G -Xms1g -Xss512M"
  scala ${BinPath}/java_abstract-1.0-jar-with-dependencies.jar ${ConfigAbstract}
}

_train() {
  set -x
  onmt_train -config ${ConfigFile} -log_file ${LogFile}
}

_preprocess() {
  set -x
  onmt_preprocess -config ${ConfigFile} -log_file ${LogFile}
}

_translate() {
  beam=$1
  onmt_translate -config ${ConfigFile} \
                 -model ${ModelCheckpoint} \
                 -output ${TranslateOutput} \
                 -tgt ${TranslateTarget} \
                 -src ${TranslateSource} \
                 -log_file ${LogFile} \
                 -beam_size ${beam}
}

inference(){
  echo "------------------- TESTING BEAM SEARCH ------------------------"
  beam_widths=("5" "10" "15" "20" "25" "30" "35" "40" "45" "50" "100" "200")

  for beam_width in ${beam_widths[*]}
  do
    echo "Beam width: $beam_width"
    SECONDS=0
    _translate ${$beam_width}
    elapsed=$SECONDS
    echo "---------- TIME REPORT ----------"
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

   *)
     echo "There is no match case for ${dataset}"
     echo "Usage: $0 dataset[small|small_old|small_tree|small_path] " >&2
     exit 1
   ;;
esac