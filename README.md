# colmap_custom_matches
Extract features, match features, put matches into COLMAP pipeline.

# Dependencies:
Install `requirements.txt`. Install [COLMAP](https://github.com/colmap/colmap).

# Usage:
First, generate data: \
`python main.py --input /data/Datasets/4seasons/four_seasons_snippet/recording_2020-04-07_11-33-45/distorted_images/cam0 --calibration /data/Datasets/4seasons/four_seasons_snippet/calibration/calib_0.txt --build_dir ~/colmap_custom_matches/test_snippet`

Second, create colmap db: \
`colmap database_creator --database_path test_snippet/database.db` 

Third, popoulate colmap db: \
`python colmap_import.py --path_image_pairs ~/colmap_custom_matches/test_snippet/matches/match_list.txt --dir_matches ~/colmap_custom_matches/test_snippet/matches --path_database ~/colmap_custom_matches/test_snippet/database.db --dir_images /data/Datasets/4seasons/four_seasons_snippet/recording_2020-04-07_11-33-45/distorted_images/cam0`

Fourth, profit?

NOTE: this currently only works for a fisheye calibration whose `calibration.txt` is in the same format as the [4Seasons](https://cvg.cit.tum.de/data/datasets/4seasons-dataset)