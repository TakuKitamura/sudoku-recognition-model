STEP=8025
INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=/Users/kitamurataku/work/sudoku-recognition-model/tf/models/model/ssd_mobilenet_v1_sudoku.config
TRAINED_CKPT_PREFIX=/Users/kitamurataku/work/sudoku-recognition-model/tf/models/model/model.ckpt-${STEP}
EXPORT_DIR=/Users/kitamurataku/work/sudoku-recognition-model/tf/models/model/frozenModel/sudoku-${STEP}
python3 /Users/kitamurataku/work/tensorflow/models/research/object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}