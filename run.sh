#!/bin/bash

CurrentDate=$(date +%F)

target=$1

# Root envs
RootPath=`pwd`/examples/learning_fix
ConfigPath=${RootPath}/config
LogFile=${RootPath}/logs/${target}-${CurrentDate}.log
BinPath=${RootPath}/bin

if [ "$#" -ne 1 ] ; then
  echo "Missing Parameters ..."
  echo "Usage: $0 target[abstract|preprocess|train|translate|all]" >&2
  exit 1
fi


############################
ConfigAbstract=${ConfigPath}/application_small.conf
ConfigFile=${ConfigPath}/small_1.yml
ModelCheckpoint=${RootPath}/data/small/small_step_20000.pt
PredictOutput=${RootPath}/data/small/predictions.txt



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
  set -x
  onmt_translate -config ${ConfigFile} -model ${ModelCheckpoint} -output ${PredictOutput} -log_file ${LogFile}
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
      _translate
   ;;

   "all")
      _preprocess
      _train
      _translate
   ;;

   *)
     echo "There is no match case for ${dataset}"
     echo "Usage: $0 dataset[small|small_old|small_tree|small_path] " >&2
     exit 1
   ;;
esac