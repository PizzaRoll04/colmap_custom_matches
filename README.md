# colmap_custom_matches
Extract features, match features, put matches into COLMAP pipeline.

# Dependencies:
Install `requirements.txt`. Install [COLMAP](https://github.com/colmap/colmap).

# Usage:
### First, generate data: \
`python main.py --input /data/Datasets/4seasons/four_seasons_snippet/recording_2020-04-07_11-33-45/distorted_images/cam0 --calibration /data/Datasets/4seasons/four_seasons_snippet/calibration/calib_0.txt --build_dir ~/colmap_custom_matches/test_snippet`

### Second, create colmap db: \
`colmap database_creator --database_path test_snippet/database.db` 

### Third, popoulate colmap db: \
`python colmap_import.py --path_image_pairs ~/colmap_custom_matches/test_snippet/matches/match_list.txt --dir_matches ~/colmap_custom_matches/test_snippet/matches --path_database ~/colmap_custom_matches/test_snippet/database.db --dir_images /data/Datasets/4seasons/four_seasons_snippet/recording_2020-04-07_11-33-45/distorted_images/cam0 --dir_keypoints ~/colmap_custom_matches/test_snippet/kpts` 

### Fourth, run geometric verification: \
`colmap matches_importer --database_path test_snippet/database.db --match_list_path test_snippet/image-pairs.txt --match_type pairs` \
\
You can also use `colmap gui` and navigate to `Processing -> Feature matching` then select `Custom` tab. Select `Image pairs` for the match type and select the `path/to/image-pairs.txt` for the match path then click `Run`.

### Fifth, import translation priors into SQL db \
I don't do this in this repo

### Sixth, sparse reconstruction: \
`colmap mapper --database_path test_snippet/database.db --image_path /data/Datasets/4seasons/four_seasons_snippet/recording_2020-04-07_11-33-45/distorted_images/cam0 --output_path test_snippet`

# Notes:
- In order to use GPU acceleration with SIFT, you have to build opencv from source with cuda enabled.
- This currently only works for a fisheye calibration whose `calibration.txt` is in the same format as the [4Seasons](https://cvg.cit.tum.de/data/datasets/4seasons-dataset)