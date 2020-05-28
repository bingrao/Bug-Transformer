#!/bin/bash

if [ "$#" -ne 1 ] ; then
  echo "Missing Parameters ..."
  echo "Usage: $0 dataset[small|median] " >&2
  exit 1
fi
CurrentDate=$(date +%F)

dataset=$1

# Root envs
export RootPath=`pwd`/examples/learning_fix/
export PYTHONPATH=${PYTHONPATH}:${RootPath}

case ${dataset} in
  "small")
    set -x
    export JAVA_OPTS="-Xmx32G -Xms1g -Xss512M"
    scala ${RootPath}/bin/java_abstract-1.0-jar-with-dependencies.jar ${RootPath}/config/application_small.conf
  ;;

  "median")
      set -x
    export JAVA_OPTS="-Xmx32G -Xms1g -Xss512M"
    scala ${RootPath}/bin/java_abstract-1.0-jar-with-dependencies.jar ${RootPath}/config/application_median.conf
   ;;

   *)
     echo "There is no match case for ${dataset}"
     echo "Usage: $0 dataset[small|median] " >&2
     exit 1
  ;;
esac
