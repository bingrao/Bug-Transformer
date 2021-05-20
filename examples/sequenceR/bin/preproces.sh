#!/usr/bin/env bash
ROOT_PATH=`pwd`/../
BLEU_CMD=${ROOT_PATH}/bin/multi-bleu.perl
BUGGY_PATH=${ROOT_PATH}/data/small/raw/buggy
FIXED_PATH=${ROOT_PATH}/data/small/raw/fixed
BUGGY_CODE=${ROOT_PATH}/data/small/raw/buggy.txt
FIXED_CODE=${ROOT_PATH}/data/small/raw/fixed.txt

[ -f ${BUGGY_CODE} ] && rm -fr ${BUGGY_CODE}
[ -f ${FIXED_CODE} ] && rm -fr ${FIXED_CODE}


buggy_cnt=$(ls ${BUGGY_PATH} | wc -l)
fixed_cnt=$(ls ${FIXED_PATH} | wc -l)

if [ $buggy_cnt != $fixed_cnt ]
then
   echo "The total number does not match ${buggy_cnt} != ${fixed_cnt}"
   exit 1
else
   echo "The total file is ${buggy_cnt}"
fi 

for i in $(seq 1 $buggy_cnt)
do
   [ $((i % 5000)) == 1 ] && echo "Step $i"
   cat ${BUGGY_PATH}/${i}.java |  tr '\n' ' ' | tr -s " " >>${BUGGY_CODE}
   echo $'\r' >>${BUGGY_CODE}
   
   cat ${FIXED_PATH}/${i}.java |  tr '\n' ' ' | tr -s " " >>${FIXED_CODE}
   echo $'\r' >>${FIXED_CODE}
done

echo "************************* BLEU ************************"
${BLEU_CMD} ${BUGGY_CODE} < ${FIXED_CODE}