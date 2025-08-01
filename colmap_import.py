# Import the features and matches into a COLMAP database.
#
# Copyright 2017: Johannes L. Schoenberger <jsch at inf.ethz.ch>
# modified by Nico Buono <nbuono@mit.edu>

import argparse
import numpy as np
import sqlite3
import struct

from pathlib import Path


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        return 2147483647 * image_id2 + image_id1
    else:
        return 2147483647 * image_id1 + image_id2
    
def read_matrix(path, dtype):
    with open(path, "rb") as fid:
        shape = np.fromfile(fid, count=2, dtype=np.int32)
        matrix = np.fromfile(fid, count=shape[0] * shape[1], dtype=dtype)
    return matrix.reshape(shape)    
    
def main():
    parser = argparse.ArgumentParser(description="Process some images.")
    parser.add_argument("--path_image_pairs", type=str, required=True, help="Path to image-pairs.txt")
    parser.add_argument("--dir_matches", type=str, required=True, help="Path to directory of matches")
    parser.add_argument("--path_database", type=str, required=True, help="Path to database.db")
    parser.add_argument("--dir_images", type=str, required=True, help="Path to directory of images")
    parser.add_argument("--dir_keypoints", type=str, required=True, help="Path to directory of keypoints")

    args = parser.parse_args()

    path_image_pairs = Path(args.path_image_pairs)
    dir_matches = Path(args.dir_matches)
    path_database = Path(args.path_database)
    dir_images = Path(args.dir_images)
    dir_kpts = Path(args.dir_keypoints)
    
    connection = sqlite3.connect(path_database)
    cursor = connection.cursor()

    # wipe the db
    cursor.execute("DELETE FROM keypoints;")
    cursor.execute("DELETE FROM descriptors;")
    cursor.execute("DELETE FROM matches;")
    cursor.execute("DELETE FROM cameras")
    cursor.execute("DELETE FROM images")
    connection.commit()

    ### write camera to db
    cam_id = 1
    params = [
        527.9990706330082,
        527.963495807245,
        399.18451401412665,
        172.8193108347693,
        -0.03559759964255725,
        -0.005093721310999416,
        0.019716282737702494,
        -0.01583280039499382,
    ]

    # pack as binary double
    params_blob = struct.pack("<" + "d" * len(params), *params)
    # cal_4seasons_opencv = (
    #     "527.9990706330082 527.963495807245 399.18451401412665 "
    #     "172.8193108347693 -0.03559759964255725 -0.005093721310999416 "
    #     "0.019716282737702494 -0.01583280039499382"
    # )
    cursor.execute(
        "INSERT INTO cameras (camera_id, model, width, height, params, prior_focal_length) VALUES (?, ?, ?, ?, ?, ?)",
        (1, 5, 800, 400, params_blob, 0)
    )

    # cursor.execute(
    #     f"""
    #     INSERT INTO cameras (camera_id, model, width, height, params, prior_focal_length)
    #     VALUES ({cam_id}, 5, 800, 400, '{cal_4seasons_opencv}', 0)
    #     """
    # )
    connection.commit()

    ### write images to db
    for img_path in sorted(dir_images.glob("*.png")):
        cursor.execute("INSERT INTO images (name, camera_id) VALUES (?, ?)", (img_path.name, cam_id))
    connection.commit()

    ### get images and names from db
    images = {}
    cursor.execute("SELECT name, image_id FROM images;")
    for row in cursor:
        images[row[0]] = row[1]

    print(images)

    ### write keypoints to db
    for file_img_kpts in dir_kpts.glob("*.txt"):
        keypoints = []
        with open(file_img_kpts, "r") as f_img_kpts:
            for line in f_img_kpts:
                data_line = line.strip().split()
                if len(data_line):
                    x, y = float(data_line[0]), float(data_line[1])
                    keypoints.append([x,y])
        keypoints = np.array(keypoints, dtype=np.float32)
        assert keypoints.shape[1] == 2
        keypoints_bin = keypoints.tobytes()
        image_id = images[file_img_kpts.name.rstrip(".txt")]
        cursor.execute(
            "INSERT INTO keypoints (image_id, rows, cols, data) VALUES (?, ?, ?, ?);",
            (image_id, keypoints.shape[0], keypoints.shape[1], keypoints_bin)
        )

    ### write matches to db
    image_pairs = []
    with open(path_image_pairs, "r") as f_image_pairs:
        for image_pair in f_image_pairs:
            image_name1, image_name2 = image_pair.strip().split()
            image_pairs.append((image_name1, image_name2))
            print("Importing matches for ", image_name1, " ", image_name2)
            image_id1, image_id2 = images[image_name1], images[image_name2]
            image_pair_id = image_ids_to_pair_id(image_id1, image_id2)

            path_match_txt = dir_matches / f"{image_name1}-{image_name2}.txt"
            idxs_img_1 = []
            idxs_img_2 = []
            with open(path_match_txt, "r") as f_match:
                for match_pair in f_match:
                    idx_1, idx_2 = match_pair.strip().split()
                    idxs_img_1.append(idx_1)
                    idxs_img_2.append(idx_2)
            matches = np.hstack((
                np.array(idxs_img_1, dtype=np.uint32).reshape(-1, 1),
                np.array(idxs_img_2, dtype=np.uint32).reshape(-1, 1)
            ))
            assert matches.shape[1] == 2
            matches_bit = matches.tobytes()
            print(matches.shape)
            cursor.execute(
                "INSERT INTO matches (pair_id, rows, cols, data) VALUES (?, ?, ?, ?);",
                (image_pair_id, matches.shape[0], matches.shape[1], matches_bit)
            )
    connection.commit()

    ### write image pairs txt
    path_image_pairs = path_database.parent / "image-pairs.txt"
    # path_image_pairs.mkdir(parents=True, exist_ok=True)
    with open(path_image_pairs, "w") as fid:
        for image_name1, image_name2 in image_pairs:
            fid.write("{} {}\n".format(image_name1, image_name2))

    cursor.close()
    connection.close()

if __name__ == "__main__":
    main()