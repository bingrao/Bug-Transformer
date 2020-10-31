#!/usr/bin/env bash

dataset="small"
config_index=10
step=100000
nums_worker=64

############################# Root envs ############################
RootPath=$(pwd)
ProjectPath=${RootPath}/examples/learning_fix
ConfigPath=${ProjectPath}/config/${dataset}; [ -d "$ConfigPath" ] || mkdir -p "$ConfigPath"
BinPath=${ProjectPath}/bin; [ -d "$BinPath" ] || mkdir -p "$BinPath"

LogPath=${ProjectPath}/logs; [ -d "$LogPath" ] || mkdir -p "$LogPath"
DataPath=${ProjectPath}/data; [ -d "$DataPath" ] || mkdir -p "$DataPath"
CurrentDate=$(date +%F)
prefix="performance-analysis"
LogFile=${LogPath}/${CurrentDate}-${prefix}.log

FIXED_PATH=${ProjectPath}/data/${dataset}/test-fixed.txt
BUGGY_PATH=${ProjectPath}/data/${dataset}/test-buggy.txt



function logInfo() {
    echo "[$(date +"%F %T,%3N") INFO] $1" | tee -a "${LogFile}"
}


function _performance() {

  n_best=$1
  measure=$2
  logInfo "-------------------- Performance Analysis for dataset[${dataset}], config[${config_index}], train step[${step}] --------------------"
  OUTPUT_DIR=${ProjectPath}/data/${dataset}/${config_index}/${step}/${measure}/; [ -d "$OUTPUT_DIR" ] || mkdir -p "$OUTPUT_DIR"
  PREDT_PATH=${ProjectPath}/data/${dataset}/${config_index}/${step}/predictions_${n_best}_${n_best}.txt
  case ${measure} in
    "similarity")
      # shellcheck disable=SC2091
      python "${BinPath}"/performance_analysis.py \
        -output="${OUTPUT_DIR}" \
        -src_buggy="${BUGGY_PATH}" \
        -src_fixed="${FIXED_PATH}" \
        -pred_fixed="${PREDT_PATH}" \
        -project_log="${LogFile}" \
        -n_best="${n_best}" \
        -log4j_config="${ConfigPath}"/log4j.properties \
        -jar="${BinPath}"/java_abstract-1.0-jar-with-dependencies.jar \
        -measure="${measure}" | tee -a "${LogFile}"
    ;;

    "ast")
      export JAVA_OPTS="-Xmx32G -Xms1g -Xss512M -Dlog4j.configuration=file:///${ConfigPath}/log4j.properties"
      scala "${BinPath}"/java_abstract-1.0-jar-with-dependencies.jar -run_type "astdiff" \
        -buggy_path "${BUGGY_PATH}" \
        -fixed_path "${FIXED_PATH}" \
        -predt_path "${PREDT_PATH}" \
        -n_best "${n_best}" \
        -nums_worker "${nums_worker}" \
        -measure="${measure}" \
        -output_dir "${OUTPUT_DIR}" | tee -a "${LogFile}"
    ;;

    "bleu")
      # shellcheck disable=SC2091
      python "${BinPath}"/performance_analysis.py \
        -output="${OUTPUT_DIR}" \
        -src_buggy="${BUGGY_PATH}" \
        -src_fixed="${FIXED_PATH}" \
        -pred_fixed="${PREDT_PATH}" \
        -project_log="${LogFile}" \
        -n_best="${n_best}" \
        -log4j_config="${ConfigPath}"/log4j.properties \
        -jar="${BinPath}"/java_abstract-1.0-jar-with-dependencies.jar \
        -measure="${measure}" | tee -a "${LogFile}"
    ;;

    *)
      logInfo "There is no match case for ${measure}"
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

    logInfo "[BLEU] buggy vs fixed"
    "${BinPath}"/multi-bleu.perl "${FIXED_BEST_OUTPUT}" < "${BUGGY_BEST_OUTPUT}" | tee -a "${LogFile}"

    logInfo "[BLUE] predt vs fixed"
    "${BinPath}"/multi-bleu.perl "${FIXED_BEST_OUTPUT}" < "${PREDT_BEST_OUTPUT}" | tee -a "${LogFile}"

  else
    [  -f "${PREDT_BEST_OUTPUT}" ] || logInfo "The input ${PREDT_BEST_OUTPUT} does not exist ..."
    [  -f "${FIXED_BEST_OUTPUT}" ] || logInfo "The input ${FIXED_BEST_OUTPUT} does not exist ..."
    [  -f "${BUGGY_BEST_OUTPUT}" ] || logInfo "The input ${BUGGY_BEST_OUTPUT} does not exist ..."
  fi
}


n_bests=("1" "5" "10" "15" "20" "25" "30" "35" "40" "45" "50")
for n_best in ${n_bests[*]}
do
#  _performance "${n_best}" "bleu"
#  _performance "${n_best}" "similarity"
 _performance "${n_best}" "ast"
done
#
# _performance 1 "ast"
