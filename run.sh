#!/usr/bin/env bash


############################# Root envs ############################
RootPath=$(pwd)
ProjectPath=${RootPath}/examples/learning_fix
ConfigPath=${ProjectPath}/config; [ -d "$ConfigPath" ] || mkdir -p "$ConfigPath"
BinPath=${ProjectPath}/bin; [ -d "$BinPath" ] || mkdir -p "$BinPath"
LogPath=${ProjectPath}/logs; [ -d "$LogPath" ] || mkdir -p "$LogPath"
DataPath=${ProjectPath}/data; [ -d "$DataPath" ] || mkdir -p "$DataPath"
CurrentDate=$(date +%F)

function help() {
     echo "Usage: [export CUDA_VISIBLE_DEVICES=0;] $0 dataset[small|median|big|small_old] target[abstract|preprocess|train|translate|all|inference] configFile" >&2
     echo "Example: Using third (or first by default) GPU to train small dataset with small_1.yml config file"
     echo "       - export CUDA_VISIBLE_DEVICES=2; bash run.sh small train small_1.yml"
     echo "       - bash run.sh small train small_1.yml"
}

if [ "$#" -ne 3 ] ; then
  echo "Missing Parameters ..."
  help
  exit 1
fi

########################### Project Parameters #######################
dataset=$1
target=$2
configFile=$3
prefix="${dataset}-$target-$(echo "${configFile}" | cut -d'.' -f1)"

echo "[$(date +"%F %T,%3N") INFO] Check Config file \"$configFile\" if match regex [*_[0-9]+\.yml], for example small_1.yml"
$(echo "$configFile" | grep -Eq  '*_[0-9]+\.yml'$) || exit 1
echo "[$(date +"%F %T,%3N") INFO] Config file name is good"

config_index=$(echo "${configFile}" |  tr -dc '0-9')

DataOutputPath=${DataPath}/${dataset}/${config_index}; [ -d "$DataOutputPath" ] || mkdir -p "$DataOutputPath"

# Log file
LogFile=${LogPath}/${CurrentDate}-${prefix}.log
#TensorboardDir=${LogPath}/${CurrentDate}-${prefix}; [ -d "$TensorboardDir" ] || mkdir -p "$TensorboardDir"

function logInfo() {
    echo "[$(date +"%F %T,%3N") INFO] $1" | tee -a "${LogFile}"
}

# Config files for model data preprocess, train, translate
ConfigFile=${ConfigPath}/${configFile}
if [ -f "$ConfigFile" ]; then
    logInfo "Loading config from $ConfigFile."
else
    logInfo "Config file $ConfigFile does not exist."
    exit 1
fi

function parse_and_print_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}

function parse_yaml() {
   local config_file=$1
   local target=$2
   local parameter=$3
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   local reg=$(sed -ne "s|^\($s\):|\1|" \
          -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
          -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  "$config_file" |
           awk -F$fs '{
              indent = length($1)/2;
              vname[indent] = $2;
              for (i in vname) {if (i > indent) {delete vname[i]}}
              if (length($3) > 0) {
                 vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])}
                 if (vn == "'$target'" && $2 == "'$parameter'"){
                   print($3);
                 }
              }
           }')
   echo "${reg}"
}


######### Internal Special parameters for model translate ###################

# The buggy code (source) to translate task
TranslateSource=${RootPath}/$(parse_yaml "${ConfigFile}" "translate" "src")

# The fixed code (target) to translate task
TranslateTarget=${RootPath}/$(parse_yaml "${ConfigFile}" "translate" "tgt")

# The model predict output, each line is corresponding to the line in buggy code
[ -z "$(parse_yaml "${ConfigFile}" "translate" "output")" ] && TranslateOutput=${RootPath}/$(parse_yaml "${ConfigFile}" "translate" "output") || TranslateOutput=${DataOutputPath}/predictions.txt

[ -z "$(parse_yaml "${ConfigFile}" "train" "save_model")" ] && ModelCheckpointPrefix=${RootPath}/"$(parse_yaml "${ConfigFile}" "train" "save_model")" || ModelCheckpointPrefix=${DataOutputPath}/${dataset}

# The beam size for prediction
TranslateBeamSize=1
#logInfo "TranslateBeamSize=${TranslateBeamSize}"

#####################################################################
########################### Helper functions  #######################
#####################################################################

function _abstract() {
  logInfo "------------------- Code Abstract ------------------------"
  # Config file for scala application to generate abstract code
  ConfigAbstract=${ConfigPath}/application_${dataset}.conf
  if [ -f "$ConfigAbstract" ]; then
      logInfo "$ConfigAbstract exists."
  else
      logInfo "$ConfigAbstract does not exist."
      exit 1
  fi
  export JAVA_OPTS="-Xmx32G -Xms1g -Xss512M"
  scala "${BinPath}"/java_abstract-1.0-jar-with-dependencies.jar "${ConfigAbstract}"

  logInfo "Generated abstract code is done, then split into train, test, eval dataset ..."
  OutputBuggyDir=$(cat "${ConfigAbstract}" | grep -e "OutputBuggyDir" | awk '{print $3}' | tr -d '"' | tr -d '\r')
  OutputFixedDir=$(cat "${ConfigAbstract}" | grep -e "OutputFixedDir" | awk '{print $3}' | tr -d '"' | tr -d '\r')

  OutputBuggyFile=${OutputBuggyDir}/total/buggy.txt
  OutputFixedFile=${OutputFixedDir}/total/fixed.txt

  logInfo "Check ${OutputBuggyFile} if exist"
  [ -f "${OutputBuggyFile}" ] || exit 1

  logInfo "Check ${OutputFixedFile} if exist"
  [ -f "${OutputFixedFile}" ] || exit 1

  buggy_cnt=$(wc -l < "${OutputBuggyFile}")
  fixed_cnt=$(wc -l < "${OutputFixedFile}")

  if [ "$buggy_cnt" != "$fixed_cnt" ]
  then
     logInfo "The total number does not match ${buggy_cnt} != ${fixed_cnt}"
     exit 1
  fi

  train_cnt="$(echo "scale=0; $buggy_cnt *  0.8 / 1" | bc)"
  eval_cnt="$(echo "scale=0; $buggy_cnt *  0.12 / 1" | bc)"

  logInfo "BLUE value <buggy.txt, fixed.txt>, count: ${buggy_cnt}"
  "${BinPath}"/multi-bleu.perl "${OutputBuggyFile}" < "${OutputFixedFile}" | tee -a "${LogFile}"

  split -l "${train_cnt}" "${OutputBuggyFile}" train-buggy
  split -l "${train_cnt}" "${OutputFixedFile}" train-fixed
  mv ./train-buggyaa "${OutputBuggyDir}"/train-buggy.txt
  mv ./train-fixedaa "${OutputFixedDir}"/train-fixed.txt

  logInfo "BLUE value <train-buggy.txt, train-fixed.txt>, count: ${train_cnt}"
  "${BinPath}"/multi-bleu.perl "${OutputBuggyDir}"/train-buggy.txt < "${OutputFixedDir}"/train-fixed.txt | tee -a "${LogFile}"

  split -l "${eval_cnt}" ./train-buggyab eval-buggy; rm -fr train-buggyab
  split -l "${eval_cnt}" ./train-fixedab eval-fixed; rm -fr train-fixedab

  mv ./eval-buggyaa "${OutputBuggyDir}"/eval-buggy.txt
  mv ./eval-fixedaa "${OutputFixedDir}"/eval-fixed.txt

  logInfo "BLUE value <eval-buggy.txt, eval-fixed.txt>, count: ${eval_cnt}"
  "${BinPath}"/multi-bleu.perl "${OutputBuggyDir}"/eval-buggy.txt < "${OutputFixedDir}"/eval-fixed.txt | tee -a "${LogFile}"

  mv ./eval-buggyab "${OutputBuggyDir}"/test-buggy.txt
  mv ./eval-fixedab "${OutputFixedDir}"/test-fixed.txt
  logInfo "BLUE value <test-buggy.txt, test-fixed.txt, count: $((buggy_cnt - train_cnt - eval_cnt))>"
  "${BinPath}"/multi-bleu.perl "${OutputBuggyDir}"/test-buggy.txt < "${OutputFixedDir}"/test-fixed.txt | tee -a "${LogFile}"
}

function _train() {
  logInfo "------------------- Training ------------------------"
  # The numbers of GPU nodes used for training task
  Nums_GPU=$(parse_yaml "${ConfigFile}" "train" "world_size")
  logInfo "Using ${Nums_GPU} for training task ... "
  [[ -z "${CUDA_VISIBLE_DEVICES}" ]] && export CUDA_VISIBLE_DEVICES=$(seq -s, 0 "${Nums_GPU}") || logInfo "exist: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  logInfo "The checkpoint will be saved with prefix ${ModelCheckpointPrefix}"
  onmt_train -save_model "${ModelCheckpointPrefix}" -config "${ConfigFile}" -log_file "${LogFile}"
  #-tensorboard -tensorboard_log_dir "${TensorboardDir}"
}

function _preprocess() {
  logInfo "------------------- Preprocess  ------------------------"
  onmt_preprocess -config "${ConfigFile}" -log_file "${LogFile}"
}

function _translate() {
  logInfo "------------------- Translate  ------------------------"
  beam_size=$1

  logInfo "Beam Size ${beam_size}"

  # Test if checkpoint is set up in the config file
  if [ -z "$(parse_yaml "${ConfigFile}" "translate" "model")" ]
  then
      logInfo "The Checkpoint model is not be set up in ${ConfigFile}, then search best one."

      # NF means the number of parameters in awk command
      ModelCheckpoint=$(ls ${ModelCheckpointPrefix}-step-*.pt |
          awk -F '-' 'BEGIN{maxv=-1000000} {score=$(NF-4); if (score > maxv) {maxv=score; max=$0}}  END{ print max}')
  else
      ModelCheckpoint=${RootPath}/$(parse_yaml "${ConfigFile}" "translate" "model")
  fi

  logInfo "Loading checkpoint ${ModelCheckpoint} for translate job ..."
  logInfo "The output prediction will be save to ${TranslateOutput}"
  onmt_translate -config "${ConfigFile}" -log_file "${LogFile}" -beam_size "${beam_size}" -model "${ModelCheckpoint}" -output "${TranslateOutput}"

  # Backup all predictions txt
  step=$(echo "${ModelCheckpoint}" | awk -F'/' '{print $NF}' | cut -d'-' -f 3)
  DataOutputStepPath=${DataOutputPath}/${step}; [ -d "$DataOutputStepPath" ] || mkdir -p "$DataOutputStepPath"
  cp "${TranslateOutput}" "${DataOutputStepPath}"/predictions_"${beam_width}".txt
}

function _evaluation() {
  logInfo "------------------- Bleu ------------------------"

  logInfo "buggy vs fixed"
  "${BinPath}"/multi-bleu.perl "${TranslateTarget}" < "${TranslateSource}" | tee -a "${LogFile}"

  logInfo "prediction vs fixed"
  "${BinPath}"/multi-bleu.perl "${TranslateTarget}" < "${TranslateOutput}" | tee -a "${LogFile}"

  logInfo "------------------- Classification ------------------------"

  total=$(wc -l "${TranslateTarget}" | awk '{print $1}')
  logInfo "Test Set: $total"

  output=$(python3 "${BinPath}"/prediction_classifier.py "${TranslateSource}" "${TranslateTarget}" "${TranslateOutput}" 2>&1)
  perf=$(awk '{print $1}' <<< "$output")
  changed=$(awk '{print $2}' <<< "$output")
  bad=$(awk '{print $3}' <<< "$output")
  perf_perc="$(echo "scale=2; $perf * 100 / $total" | bc)"

  logInfo "Perf: $perf ($perf_perc%)"
  logInfo "Pot : $changed"
  logInfo "Bad : $bad"
}

function _inference(){
  logInfo "------------------- Inference Search ------------------------"
  beam_widths=("1" "5" "10" "15" "20" "25" "30" "35" "40" "45" "50" "100")
  for beam_width in ${beam_widths[*]}
  do
    logInfo "******************** Beam width: $beam_width ********************"
    SECONDS=0
    _translate "${beam_width}"
    elapsed=$SECONDS
    logInfo "---------- Time report ----------"
	  logInfo "Total seconds: $elapsed"

    total=$(wc -l "${TranslateSource}" | awk '{print $1}')
    patches=$((total * beam_width))
    avg="$(echo "scale=6; $elapsed / $patches" | bc)"
    avg_bug="$(echo "scale=6; $elapsed / $total" | bc)"

    logInfo "Total bugs: $total"
    logInfo "Total patches: $patches"
    logInfo "Avg patch/sec: $avg"
    logInfo "Avg bug/sec: $avg_bug"

    _evaluation

    rm -fr "${TranslateOutput}"
    printf "\n\n" | tee -a "${LogFile}"
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
      _evaluation
   ;;

   "all")
      _train
      _inference
   ;;

   "inference")
      _inference
   ;;

   *)
     logInfo "There is no match case for ${target}"
     help
     exit 1
   ;;
esac
