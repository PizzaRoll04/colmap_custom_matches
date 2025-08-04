#!/bin/bash

set -e

PATH_DATA=$1
PATH_CALIBRATION=$2
BUILD_DIR=$3
PATH_DATABASE=$4
PATH_PYTHON_VENV=$5
ROBOT_PATH=$6
BOOL_GENERATE_DATA=$7

PATH_IMAGES=$PATH_DATA/distorted_images/cam0

echo "Data path: $PATH_DATA"
echo "Calibration path: $PATH_CALIBRATION"
echo "Build dir: $BUILD_DIR"
echo "Database path: $PATH_DATABASE"
echo "Python venv path: $PATH_PYTHON_VENV"
echo "Robot path: $ROBOT_PATH"
echo ""
echo "Images path: $PATH_IMAGES"
echo "Bool generate data: $BOOL_GENERATE_DATA"

source $PATH_PYTHON_VENV/bin/activate

echo ""
if [[ "$BOOL_GENERATE_DATA" == "true" ]]; then
    echo "Generating Data"
    python main.py --input $PATH_IMAGES\
    --calibration $PATH_CALIBRATION --build_dir $BUILD_DIR
elif [[ "$BOOL_GENERATE_DATA" == "false" ]]; then
    echo "Skipping data generation"
else
    echo "Invalid boolean: $BOOL_GENERATE_DATA   (must be 'true' or 'false')" >&2
    exit 1
fi

colmap database_creator --database_path $PATH_DATABASE
echo "Generated db"

echo "Importing data into db"
python colmap_import.py --path_image_pairs $BUILD_DIR/matches/match_list.txt --dir_matches $BUILD_DIR/matches\
 --path_database $PATH_DATABASE --dir_images $PATH_IMAGES --dir_keypoints $BUILD_DIR/kpts

echo "Two view verification"
colmap matches_importer --database_path $PATH_DATABASE --match_list_path $BUILD_DIR/image-pairs.txt --match_type pairs

# original_dir="$PWD"
# echo "original_dir: $original_dir"
# cd $ROBOT_PATH
# echo "Importing pose priors"
# bazel run //experimental/learn_descriptors:write_to_colmap_database -- --data_dir $PATH_DATA\
#  --calibration_dir $PATH_CALIBRATION --colmap_database_path $PATH_DATABASE
# echo "Heartbeat"
# cd $original_dir
