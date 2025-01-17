#!/usr/bin/env bash

dataset=$1
target=$2
configFile=$3

############################# Root envs ############################
RootPath=$(pwd)
ProjectPath=${RootPath}/examples/codebert
ConfigPath=${ProjectPath}/config/${dataset}; [ -d "$ConfigPath" ] || mkdir -p "$ConfigPath"
BinPath=${ProjectPath}/bin; [ -d "$BinPath" ] || mkdir -p "$BinPath"
LogPath=${ProjectPath}/logs; [ -d "$LogPath" ] || mkdir -p "$LogPath"
DataPath=${ProjectPath}/data; [ -d "$DataPath" ] || mkdir -p "$DataPath"
CurrentDate=$(date +%F)

function help() {
     echo "Usage: [export CUDA_VISIBLE_DEVICES=0;] $0 dataset target configFile" >&2
     echo "       dataset: [small|median|big]"
     echo "       target:  [train|translate|all|inference|performance|loop_translate]"
     echo "Example: Using third (or first by default) GPU to train small dataset with small_1.yml config file"
     echo "Example: The default direcotry that system searches config files: ${ProjectPath}/config/[small|median|big]"
     echo "       - export CUDA_VISIBLE_DEVICES=2,3; bash run_codebert.sh small train small_1.yml"
     echo "       - bash run_codebert.sh small train small_1.yml"
}

if [ "$#" -ne 3 ] ; then
  echo "Missing Parameters ..."
  help
  exit 1
fi

########################### Project Parameters #######################

prefix="${dataset}-$target-$(echo "${configFile}" | cut -d'.' -f1)"

config_index=$(echo "${configFile}" |  tr -dc '0-9')

# Log file
LogFile=${LogPath}/${CurrentDate}-${prefix}.log

DataOutputPath=${DataPath}/${dataset}/${config_index}; [ -d "$DataOutputPath" ] || mkdir -p "$DataOutputPath"

#######################################################################################################
######################################## Helper functions  ############################################
#######################################################################################################

function logInfo() {
    echo "[$(date +"%F %T,%3N") INFO] $1" | tee -a "${LogFile}"
}

logInfo "Check Config file \"${configFile}\" -------- Pass"

# Config files for model data preprocess, train, translate
ConfigFile=${ConfigPath}/${configFile}
if [ -f "$ConfigFile" ]; then
    logInfo "Loading config from $ConfigFile."
else
    logInfo "Config file $ConfigFile does not exist."
    exit 1
fi

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

function parse_one_layer_yaml() {
   local config_file=$1
   local target=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   local reg=$(sed -ne "s|^\($s\):|\1|" \
          -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
          -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  "$config_file" |
           awk -F$fs '{
              indent = length($1)/2;
              vname[indent] = $2;
              for (i in vname) {if (i > indent) {delete vname[i]}}
              if (length($2) > 0) {
                 vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])}
                 if (vn == "'$target'"){
                   print($2);
                 }
              }
           }')
   echo "${reg}"
}


function _train() {
  logInfo "------------------- Training ------------------------"
  python ${BinPath}/train.py -config "${ConfigFile}" -log_file "${LogFile}" --output_dir "${DataOutputPath}"
}

function _translate() {

  beam_size=$1
  n_best=$2
  TranslateBestRatio=$3
  enableTranslate=true  # Available options: true, false
  measure='similarity'  # The measure using in classification task, avaialble options: 'similarity', 'ast', 'bleu'
  nums_thread=32  # Nums of thread to conduct classification tasks
  logInfo "------------------- Translate  ------------------------"
  ######### Internal Special parameters for model translate ###################
  SECONDS=0

  # The buggy code (source) to translate task
  TranslateSource=$(parse_yaml "${ConfigFile}" "translate" "src")
  # The fixed code (target) to translate task
  TranslateTarget=$(parse_yaml "${ConfigFile}" "translate" "tgt")

  # The model predict output, each line is corresponding to the line in buggy code
  TranslateOutputDir=$(parse_yaml "${ConfigFile}" "translate" "output")
  [ -z "${TranslateOutputDir}" ] && TranslateOutputDir=${DataOutputPath}/pred_${beam_size} || TranslateOutputDir=${RootPath}/${TranslateOutput}

  logInfo "Beam Size ${beam_size}, nums of best ${n_best}"

  PredBackupPath="${TranslateOutputDir}"/predictions_"${beam_size}"_"${n_best}".txt
  PredBestPath="${TranslateOutputDir}"/predictions_"${beam_size}"_"${n_best}"_best.txt

  logInfo "The output prediction will be save to ${TranslateOutputDir}"
  python ${BinPath}/translate.py -config "${ConfigFile}" -log_file "${LogFile}" --beam_size "${beam_size}" --output_dir "${TranslateOutputDir}"

  # Backup all predictions txt
  logInfo "Backup the model output to ${PredBackupPath}"
  cp "${TranslateOutputDir}/test_0.output" "${PredBackupPath}"


  logInfo "------------------- Classification ------------------------"
  total=$(awk 'END{print NR}' "${TranslateTarget}" | awk '{print $1}')
  logInfo "Get best prediction results using n_best [${n_best}], best_ratio [${TranslateBestRatio}], measure [${measure}], nums_thread [${nums_thread}]"

  output=$(python "${BinPath}"/split_predictions.py \
      -output="${PredBestPath}" \
      -src_buggy="${TranslateSource}" \
      -src_fixed="${TranslateTarget}" \
      -pred_fixed="${PredBackupPath}" \
      -project_log="${LogFile}" \
      -n_best="${n_best}" \
      -best_ratio="${TranslateBestRatio}" \
      -log4j_config="${ConfigPath}"/log4j.properties \
      -jar="${BinPath}"/code2abs-1.0-jar-with-dependencies.jar \
      -nums_thread=${nums_thread} \
      -measure=${measure} 2>&1)

  perf=$(awk '{print $1}' <<< "$output")
  changed=$(awk '{print $2}' <<< "$output")
  bad=$(awk '{print $3}' <<< "$output")
  perf_perc="$(echo "scale=2; $perf * 100 / $total" | bc)"

  logInfo "Perf  : $perf ($perf_perc%)"
  logInfo "Pot   : $changed"
  logInfo "Bad   : $bad"
  logInfo "--------------------"
  logInfo "Total : $total"


  logInfo "------------------- Bleu ------------------------"
  logInfo "buggy vs fixed"
  "${BinPath}"/multi-bleu.perl "${TranslateTarget}" < "${TranslateSource}" | tee -a "${LogFile}"

  logInfo "prediction vs fixed"
  "${BinPath}"/multi-bleu.perl "${TranslateTarget}" < "${PredBestPath}" | tee -a "${LogFile}"


  elapsed=$SECONDS
  logInfo "---------- Time report ----------"
	logInfo "Total seconds: $elapsed"

  total=$(awk 'END{print NR}' "${TranslateSource}" | awk '{print $1}')
  patches=$((total * beam_size))
  avg="$(echo "scale=6; $elapsed / $patches" | bc)"
  avg_bug="$(echo "scale=6; $elapsed / $total" | bc)"

  logInfo "Total bugs: $total"
  logInfo "Total patches: $patches"
  logInfo "Avg patch/sec: $avg"
  logInfo "Avg bug/sec: $avg_bug"

}


function loop_translate() {

  beam_size=$1
  n_best=$2
  TranslateBestRatio=$3
  ModelCheckpoint=$4
  enableTranslate=true  # Available options: true, false
  measure='similarity'  # The measure using in classification task, available options: 'similarity', 'ast', 'bleu'
  nums_thread=32  # Nums of thread to conduct classification tasks
  logInfo "------------------- Translate  ------------------------"
  ######### Internal Special parameters for model translate ###################
  SECONDS=0

  # The buggy code (source) to translate task
  TranslateSource=${RootPath}/$(parse_yaml "${ConfigFile}" "translate" "src")

  # The fixed code (target) to translate task
  TranslateTarget=${RootPath}/$(parse_yaml "${ConfigFile}" "translate" "tgt")

  # The model predict output, each line is corresponding to the line in buggy code
  TranslateOutput=$(parse_yaml "${ConfigFile}" "translate" "output")
  [ -z "${TranslateOutput}" ] && TranslateOutput=${DataOutputPath}/predictions.txt || TranslateOutput=${RootPath}/${TranslateOutput}


  ModelCheckpointPrefix="$(parse_yaml "${ConfigFile}" "train" "save_model")"
  [ -z "${ModelCheckpointPrefix}" ] && ModelCheckpointPrefix=${DataOutputPath}/${dataset} || ModelCheckpointPrefix=${RootPath}/${ModelCheckpointPrefix}

  logInfo "Beam Size ${beam_size}, nums of best ${n_best}"

#  # Test if checkpoint is set up in the config file
#  if [ -z "$(parse_yaml "${ConfigFile}" "translate" "model")" ]
#  then
#      logInfo "The Checkpoint model is not be set up in ${ConfigFile}, then search best one."
#
#      # NF means the number of parameters in awk command
#      # shellcheck disable=SC2012
#      ModelCheckpoint=$(ls "${ModelCheckpointPrefix}"-step-*.pt |
#          awk -F '-' 'BEGIN{maxv=-1000000} {score=$(NF-4); if (score > maxv) {maxv=score; max=$0}}  END{ print max}')
#  else
#      ModelCheckpoint=${RootPath}/$(parse_yaml "${ConfigFile}" "translate" "model")
#  fi


  step=$(echo "${ModelCheckpoint}" | awk -F'/' '{print $NF}' | cut -d'-' -f 3)
  DataOutputStepPath=${DataOutputPath}/${step}; [ -d "$DataOutputStepPath" ] || mkdir -p "$DataOutputStepPath"
  PredBackupPath="${DataOutputStepPath}"/predictions_"${beam_size}"_"${n_best}".txt
  PredBestPath="${DataOutputStepPath}"/predictions_"${beam_size}"_"${n_best}"_best.txt

  if [ "${enableTranslate}" = true ]; then
    logInfo "Loading checkpoint ${ModelCheckpoint} for translate job ..."
    logInfo "The output prediction will be save to ${TranslateOutput}"
    onmt_translate -config "${ConfigFile}" -log_file "${LogFile}" -beam_size "${beam_size}" -n_best "${n_best}" -model "${ModelCheckpoint}" -output "${TranslateOutput}"

    # Backup all predictions txt
    logInfo "Backup the model output to ${PredBackupPath}"
    mv "${TranslateOutput}" "${PredBackupPath}"
  fi

  logInfo "------------------- Classification ------------------------"
  total=$(awk 'END{print NR}' "${TranslateTarget}" | awk '{print $1}')
  logInfo "Get best prediction results using n_best [${n_best}], best_ratio [${TranslateBestRatio}], measure [${measure}], nums_thread [${nums_thread}]"

  output=$(python "${BinPath}"/split_predictions.py \
      -output="${PredBestPath}" \
      -src_buggy="${TranslateSource}" \
      -src_fixed="${TranslateTarget}" \
      -pred_fixed="${PredBackupPath}" \
      -project_log="${LogFile}" \
      -n_best="${n_best}" \
      -best_ratio="${TranslateBestRatio}" \
      -log4j_config="${ConfigPath}"/log4j.properties \
      -jar="${BinPath}"/code2abs-1.0-jar-with-dependencies.jar \
      -nums_thread=${nums_thread} \
      -measure=${measure} 2>&1)

  perf=$(awk '{print $1}' <<< "$output")
  changed=$(awk '{print $2}' <<< "$output")
  bad=$(awk '{print $3}' <<< "$output")
  perf_perc="$(echo "scale=2; $perf * 100 / $total" | bc)"

  logInfo "Perf  : $perf ($perf_perc%)"
  logInfo "Pot   : $changed"
  logInfo "Bad   : $bad"
  logInfo "--------------------"
  logInfo "Total : $total"


  logInfo "------------------- Bleu ------------------------"
  logInfo "buggy vs fixed"
  "${BinPath}"/multi-bleu.perl "${TranslateTarget}" < "${TranslateSource}" | tee -a "${LogFile}"

  logInfo "prediction vs fixed"
  "${BinPath}"/multi-bleu.perl "${TranslateTarget}" < "${PredBestPath}" | tee -a "${LogFile}"


  elapsed=$SECONDS
  logInfo "---------- Time report ----------"
	logInfo "Total seconds: $elapsed"

  total=$(awk 'END{print NR}' "${TranslateSource}" | awk '{print $1}')
  patches=$((total * beam_size))
  avg="$(echo "scale=6; $elapsed / $patches" | bc)"
  avg_bug="$(echo "scale=6; $elapsed / $total" | bc)"

  logInfo "Total bugs: $total"
  logInfo "Total patches: $patches"
  logInfo "Avg patch/sec: $avg"
  logInfo "Avg bug/sec: $avg_bug"

}

function _inference() {
  TranslateBestRatio=1.0

  logInfo "------------------- Inference Search ------------------------"
  beam_widths=("15" "20" "25" "30" "35" "40" "45" "50")
#  beam_widths=("1" "5" "10" "15" "20")
  for beam_width in ${beam_widths[*]}
  do
    _translate "${beam_width}" "${beam_width}" "${TranslateBestRatio}"
    printf "\n\n" | tee -a "${LogFile}"
  done
}


function _performance() {

  n_best=$1
  measure=$2

  ModelCheckpoint=${RootPath}/$(parse_yaml "${ConfigFile}" "translate" "model")
  step=$(echo "${ModelCheckpoint}" | awk -F'/' '{print $NF}' | cut -d'-' -f 3)

  logInfo "Performance Analysis for dataset[${dataset}], config[${config_index}], train step[${step}] n_best[${n_best}}] measure[${measure}}] \t--------------------"
  OUTPUT_DIR=${ProjectPath}/data/${dataset}/${config_index}/${step}/${measure}/; [ -d "$OUTPUT_DIR" ] || mkdir -p "$OUTPUT_DIR"
  PREDT_PATH=${ProjectPath}/data/${dataset}/${config_index}/${step}/predictions_${n_best}_${n_best}.txt

  # The buggy code (source) to translate task
  TranslateSource=${RootPath}/$(parse_yaml "${ConfigFile}" "translate" "src")

  # The fixed code (target) to translate task
  TranslateTarget=${RootPath}/$(parse_yaml "${ConfigFile}" "translate" "tgt")

  case ${measure} in
    "similarity")
      # shellcheck disable=SC2091
      python "${BinPath}"/performance_analysis.py \
        -output="${OUTPUT_DIR}" \
        -src_buggy="${TranslateSource}" \
        -src_fixed="${TranslateTarget}" \
        -pred_fixed="${PREDT_PATH}" \
        -project_log="${LogFile}" \
        -n_best="${n_best}" \
        -log4j_config="${ConfigPath}"/log4j.properties \
        -jar="${BinPath}"/code2abs-1.0-jar-with-dependencies.jar \
        -measure="${measure}" | tee -a "${LogFile}"
    ;;

    "ast")
      export JAVA_OPTS="-Xmx32G -Xms1g -Xss512M -Dlog4j.configuration=file:///${ConfigPath}/log4j.properties"
      scala "${BinPath}"/code2abs-1.0-jar-with-dependencies.jar -run_type "astdiff" \
        -buggy_path "${TranslateSource}" \
        -fixed_path "${TranslateTarget}" \
        -predt_path "${PREDT_PATH}" \
        -n_best "${n_best}" \
        -nums_worker 64 \
        -measure="${measure}" \
        -output_dir "${OUTPUT_DIR}" | tee -a "${LogFile}"
    ;;

    "bleu")
      # shellcheck disable=SC2091
      python "${BinPath}"/performance_analysis.py \
        -output="${OUTPUT_DIR}" \
        -src_buggy="${TranslateSource}" \
        -src_fixed="${TranslateTarget}" \
        -pred_fixed="${PREDT_PATH}" \
        -project_log="${LogFile}" \
        -n_best="${n_best}" \
        -log4j_config="${ConfigPath}"/log4j.properties \
        -jar="${BinPath}"/code2abs-1.0-jar-with-dependencies.jar \
        -measure="${measure}" | tee -a "${LogFile}"
    ;;

    *)
      logInfo "There is no match measure case for ${measure}"
      exit 1
  esac

  PREDT_BEST_OUTPUT=${OUTPUT_DIR}/${n_best}_${measure}_predt_best.txt
  FIXED_BEST_OUTPUT=${OUTPUT_DIR}/${n_best}_${measure}_fixed_best.txt
  BUGGY_BEST_OUTPUT=${OUTPUT_DIR}/${n_best}_${measure}_buggy_best.txt

  if [ -f "${PREDT_BEST_OUTPUT}" -a -f "${FIXED_BEST_OUTPUT}" -a -f "${BUGGY_BEST_OUTPUT}" ]
  then
    err_cnt=("0" "1" "2" "3" "4" "Er")
    ERROR_OUTPUT_DIR=${OUTPUT_DIR}/${n_best}/; [ -d "$ERROR_OUTPUT_DIR" ] || mkdir -p "$ERROR_OUTPUT_DIR"
    for cnt in ${err_cnt[*]}
    do
      cat < "${PREDT_BEST_OUTPUT}" | grep -e "#[0-9]*#${cnt}#" > "${ERROR_OUTPUT_DIR}"/"${cnt}"_predt.txt
      cat < "${FIXED_BEST_OUTPUT}" | grep -e "#[0-9]*#${cnt}#" > "${ERROR_OUTPUT_DIR}"/"${cnt}"_fixed.txt
      cat < "${BUGGY_BEST_OUTPUT}" | grep -e "#[0-9]*#${cnt}#" > "${ERROR_OUTPUT_DIR}"/"${cnt}"_buggy.txt
    done

    logInfo "[BLEU] buggy vs fixed: $("${BinPath}"/multi-bleu.perl "${FIXED_BEST_OUTPUT}" < "${BUGGY_BEST_OUTPUT}")"
#    "${BinPath}"/multi-bleu.perl "${FIXED_BEST_OUTPUT}" < "${BUGGY_BEST_OUTPUT}" | tee -a "${LogFile}"

    logInfo "[BLUE] predt vs fixed: $("${BinPath}"/multi-bleu.perl "${FIXED_BEST_OUTPUT}" < "${PREDT_BEST_OUTPUT}")"
#    "${BinPath}"/multi-bleu.perl "${FIXED_BEST_OUTPUT}" < "${PREDT_BEST_OUTPUT}" | tee -a "${LogFile}"

  else
    [  -f "${PREDT_BEST_OUTPUT}" ] || logInfo "The input ${PREDT_BEST_OUTPUT} does not exist ..."
    [  -f "${FIXED_BEST_OUTPUT}" ] || logInfo "The input ${FIXED_BEST_OUTPUT} does not exist ..."
    [  -f "${BUGGY_BEST_OUTPUT}" ] || logInfo "The input ${BUGGY_BEST_OUTPUT} does not exist ..."
  fi
}

##################################################################################################
##################################### Main Function ##############################################
##################################################################################################

case ${target} in
   "train")
      _train
   ;;

   "translate")
      # The beam size for prediction
      TranslateBeamSize=$(parse_yaml "${ConfigFile}" "translate" "beam_size")
      [ -z "${TranslateBeamSize}" ] &&  TranslateBeamSize=1

      TranslateNBest=$(parse_yaml "${ConfigFile}" "translate" "n_best")
      [ -z "${TranslateNBest}" ] && TranslateNBest=1

      TranslateBestRatio=1.0
      _translate ${TranslateBeamSize} "${TranslateNBest}" "${TranslateBestRatio}"
   ;;


  "loop_translate")
      # The beam size for prediction
      TranslateBeamSize=$(parse_yaml "${ConfigFile}" "translate" "beam_size")
      [ -z "${TranslateBeamSize}" ] &&  TranslateBeamSize=1

      TranslateNBest=$(parse_yaml "${ConfigFile}" "translate" "n_best")
      [ -z "${TranslateNBest}" ] && TranslateNBest=1

      TranslateBestRatio=1.0

      for model in `ls ${DataOutputPath}/*.pt`; do
#        echo $model
        loop_translate ${TranslateBeamSize} "${TranslateNBest}" "${TranslateBestRatio}" ${model}
      done
  ;;

   "all")
      _train

       # The beam size for prediction
      TranslateBeamSize=$(parse_yaml "${ConfigFile}" "translate" "beam_size")
      [ -z "${TranslateBeamSize}" ] &&  TranslateBeamSize=1

      TranslateNBest=$(parse_yaml "${ConfigFile}" "translate" "n_best")
      [ -z "${TranslateNBest}" ] && TranslateNBest=1

      TranslateBestRatio=1.0
      _translate ${TranslateBeamSize} "${TranslateNBest}" "${TranslateBestRatio}"
   ;;

   "inference")
      _inference
   ;;

   "performance")
      n_bests=("1" "5" "10" "15" "20" "25" "30" "35" "40" "45" "50")
#      n_bests=("30" "35" "40" "45" "50")
      for n_best in ${n_bests[*]}
      do
        _performance "${n_best}" "similarity"
        _performance "${n_best}" "bleu"
#        _performance "${n_best}" "ast"
        printf "\n\n" | tee -a "${LogFile}"
      done
    ;;

   *)
     logInfo "There is no match case for ${target}"
     help
     exit 1
   ;;
esac
