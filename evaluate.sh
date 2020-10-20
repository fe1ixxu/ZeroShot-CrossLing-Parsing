LANG=$1
TYPE=multi # ["mean", "multi"]: mean -> word-level; multi -> sense-level 

PATH_TREE=PATH/FOR/TREEBANK2.6/  # e.g., data/ud-treebanks-v2.6/
MODEL_DIR=DIR/FOR/PRE-TRAINED/MODEL 
PATH_MAP=PATH/FOR/MAPING

if [ ${LANG} == en ]
then
    INPUT_FILE="${PATH_TREE}UD_English-EWT/en_ewt-ud-test.conllu"
elif [ ${LANG} == fi ]
then
    INPUT_FILE="${PATH_TREE}UD_Finnish-TDT/fi_tdt-ud-test.conllu"
elif [ ${LANG} == el ]
then
    INPUT_FILE="${PATH_TREE}UD_Greek-GDT/el_gdt-ud-test.conllu"
elif [ ${LANG} == ro ]
then
    INPUT_FILE="${PATH_TREE}UD_Romanian-RRT/ro_rrt-ud-test.conllu"
elif [ ${LANG} == pt ]
then
    INPUT_FILE="${PATH_TREE}UD_Portuguese-GSD/pt_gsd-ud-test.conllu"
elif [ ${LANG} == pl ]
then
    INPUT_FILE="${PATH_TREE}UD_Polish-LFG/pl_lfg-ud-test.conllu"
elif [ ${LANG} == es ]
then
    INPUT_FILE="${PATH_TREE}UD_Spanish-GSD/es_gsd-ud-test.conllu"
fi

MODEL_FILE="${MODEL_DIR}/model.tar.gz"
WEIGHTS_FILE="${MODEL_DIR}/best.th"
OUTPUT_FILE="${MODEL_DIR}/Sys-${LANG}-out"
OVERRIDES_PATH="allen_configs/override_embedder.jsonnet"
OVERRIDES=$(python ./src/load_overrides.py --overrides $OVERRIDES_PATH --lang $LANG --type $TYPE --mapping_path $PATH_MAP) 

CUDA_VISIBLE_DEVICES=0 python evaluate.py $MODEL_FILE $INPUT_FILE \
--output-file $OUTPUT_FILE \
--weights-file $WEIGHTS_FILE \
--cuda-device 0 \
--include-package src \
--overrides $OVERRIDES
