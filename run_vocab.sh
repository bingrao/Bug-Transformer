#!/usr/bin/env bash

dataset="small"

############################# Root envs ############################
RootPath=$(pwd)
ProjectPath=${RootPath}/examples/learning_fix
ConfigPath=${ProjectPath}/config/${dataset}; [ -d "$ConfigPath" ] || mkdir -p "$ConfigPath"
BinPath=${ProjectPath}/bin; [ -d "$BinPath" ] || mkdir -p "$BinPath"
LogPath=${ProjectPath}/logs; [ -d "$LogPath" ] || mkdir -p "$LogPath"
DataPath=${ProjectPath}/data; [ -d "$DataPath" ] || mkdir -p "$DataPath"
CurrentDate=$(date +%F)

prefix="vocabulary"

# Log file
LogFile=${LogPath}/${CurrentDate}-${prefix}.log

function logInfo() {
    echo "[$(date +"%F %T,%3N") INFO] $1" | tee -a "${LogFile}"
}

export JAVA_OPTS="-Xmx32G -Xms1g -Xss512M -Dlog4j.configuration=file:///${ConfigPath}/log4j.properties"

logInfo "------------------------- small -----------------------------"
scala "${BinPath}"/code2abs-1.0-jar-with-dependencies.jar -run_type "vocabulary" \
        -buggy_path "examples/learning_fix/data/small/raw/buggy/" \
        -fixed_path "examples/learning_fix/data/small/raw/fixed/" \
        -top_k 10000 \
        -is_abstract false | tee -a "${LogFile}"

printf "\n" | tee -a "${LogFile}"

logInfo "------------------------- small new abstraction-----------------------------"
scala "${BinPath}"/code2abs-1.0-jar-with-dependencies.jar -run_type "vocabulary" \
        -buggy_path "examples/learning_fix/data/small/total/buggy.txt" \
        -fixed_path "examples/learning_fix/data/small/total/fixed.txt" \
        -top_k 10000 \
        -is_abstract true | tee -a "${LogFile}"

printf "\n" | tee -a "${LogFile}"

logInfo "------------------------- small old abstraction-----------------------------"
scala "${BinPath}"/code2abs-1.0-jar-with-dependencies.jar -run_type "vocabulary" \
        -buggy_path "examples/learning_fix/data/small_old/total/buggy.txt" \
        -fixed_path "examples/learning_fix/data/small_old/total/fixed.txt" \
        -top_k 10000 \
        -is_abstract true | tee -a "${LogFile}"

printf "\n\n" | tee -a "${LogFile}"


logInfo "------------------------- median -----------------------------"
scala "${BinPath}"/code2abs-1.0-jar-with-dependencies.jar -run_type "vocabulary" \
        -buggy_path "examples/learning_fix/data/median/raw/buggy/" \
        -fixed_path "examples/learning_fix/data/median/raw/fixed/" \
        -top_k 10000 \
        -is_abstract false | tee -a "${LogFile}"

printf "\n" | tee -a "${LogFile}"

logInfo "------------------------- median new abstraction-----------------------------"
scala "${BinPath}"/code2abs-1.0-jar-with-dependencies.jar -run_type "vocabulary" \
        -buggy_path "examples/learning_fix/data/median/total/buggy.txt" \
        -fixed_path "examples/learning_fix/data/median/total/fixed.txt" \
        -top_k 10000 \
        -is_abstract true | tee -a "${LogFile}"

printf "\n" | tee -a "${LogFile}"

logInfo "------------------------- median old abstraction-----------------------------"
scala "${BinPath}"/code2abs-1.0-jar-with-dependencies.jar -run_type "vocabulary" \
        -buggy_path "examples/learning_fix/data/median_old/total/buggy.txt" \
        -fixed_path "examples/learning_fix/data/median_old/total/fixed.txt" \
        -top_k 10000 \
        -is_abstract true | tee -a "${LogFile}"
printf "\n\n" | tee -a "${LogFile}"


logInfo "------------------------- big src -----------------------------"
scala "${BinPath}"/code2abs-1.0-jar-with-dependencies.jar -run_type "vocabulary" \
        -buggy_path "examples/learning_fix/data/big/raw/buggy-src-total.txt" \
        -fixed_path "examples/learning_fix/data/big/raw/fixed-src-total.txt" \
        -top_k 10000 \
        -is_abstract true | tee -a "${LogFile}"
printf "\n" | tee -a "${LogFile}"

logInfo "------------------------- big new abstraction-----------------------------"
scala "${BinPath}"/code2abs-1.0-jar-with-dependencies.jar -run_type "vocabulary" \
        -buggy_path "examples/learning_fix/data/big/total/buggy.txt" \
        -fixed_path "examples/learning_fix/data/big/total/fixed.txt" \
        -top_k 10000 \
        -is_abstract true | tee -a "${LogFile}"

printf "\n" | tee -a "${LogFile}"

logInfo "------------------------- big old abstraction-----------------------------"
scala "${BinPath}"/code2abs-1.0-jar-with-dependencies.jar -run_type "vocabulary" \
        -buggy_path "examples/learning_fix/data/big_old/total/buggy.txt" \
        -fixed_path "examples/learning_fix/data/big_old/total/fixed.txt" \
        -top_k 10000 \
        -is_abstract true | tee -a "${LogFile}"