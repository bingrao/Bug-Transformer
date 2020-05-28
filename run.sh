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
ConfigPreprocess=${ConfigPath}/small_preprocess_1G_1.yml
ConfigTrain=${ConfigPath}/small_train_1G_1.yml
ConfigTranslate=${ConfigPath}/small_translate_1G_1.yml
ModelCheckpoint=${RootPath}/data/small/small_step_20000.pt

case ${target} in
   "abstract")
      set -x
      export JAVA_OPTS="-Xmx32G -Xms1g -Xss512M"
      scala ${BinPath}/java_abstract-1.0-jar-with-dependencies.jar ${ConfigAbstract}
   ;;

   "preprocess")
      set -x
      onmt_preprocess -config ${ConfigPreprocess} -log_file ${LogFile}
   ;;

   "train")
      set -x
      onmt_train -config ${ConfigTrain} -log_file ${LogFile}
   ;;

   "translate")
      set -x
      onmt_translate -config ${ConfigTranslate} -model ${ModelCheckpoint} -log_file ${LogFile}
   ;;

   "all")
      set -x
      onmt_preprocess -config ${ConfigPreprocess} -log_file ${LogFile}
      onmt_train -config ${ConfigTrain} -log_file ${LogFile}
      onmt_translate -config ${ConfigTranslate} -model ${ModelCheckpoint} -log_file ${LogFile}
   ;;

   *)
     echo "There is no match case for ${dataset}"
     echo "Usage: $0 dataset[small|small_old|small_tree|small_path] " >&2
     exit 1
   ;;
esac