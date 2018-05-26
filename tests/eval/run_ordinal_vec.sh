#!/bin/bash

hint="hint"
cipher="caesar"
v="0"

n=30000
g="0.5"
e="0.7"
ti="100"
ns="20"
repr="ordinal_vec"

t=`date '+%Y_%m_%d__%H_%M_%S'`
input_path="output/output_${hint}${cipher}cipher_v${v}/${repr}/hd_None/g_${g}/e_${e}/ti_${ti}/ns_${ns}/${n}"
output_path="output/output_${hint}${cipher}cipher_v${v}/${repr}/hd_None/g_${g}/e_${e}/ti_${ti}/ns_${ns}/${n}/eval"

mkdir -p ${output_path}/game
if [ "${hint}" = "hint" ]; then
    python3 -u -m decipher.rl -v $v --eval --input-model ${input_path}/model.bin --symbol-repr-method ${repr} --n-episode $n --gamma $g --epsilon $e --target-replace-iter $ti --n-states $ns --log-file ${output_path}/${t}.log >> ${output_path}/game/${t}.log
fi
