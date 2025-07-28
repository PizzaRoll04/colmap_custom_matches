from pathlib import Path
from typing import TypedDict
import cv2
import numpy as np

class ExtractedImagePt(TypedDict):
    img_path: Path
    keypoints: list
    descriptors: np.ndarray

class Extractor():
    image_extensions_ = (".jpg", ".jpeg", ".png")
    extracted_img_pts_ = []
    def __init__(self):
        self.extractor_ = cv2.SIFT_create()
        self.bf_matcher_ = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    def extract_images(self, directory: Path, K:np.ndarray = None, D:np.ndarray = None):
        image_files = [f for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in self.image_extensions_]
        image_files.sort()
        for image_file in image_files:
            image = cv2.imread(image_file)
            if K is not None and D is not None:
                h,w = image.shape[:2]
                image = cv2.fisheye.undistortImage(image, K, D, Knew=K, new_size=(w,h))
            keypoints, descriptors = self.extractor_.detectAndCompute(image, None)
            data : ExtractedImagePt = {"img_path": image_file, "keypoints": keypoints, "descriptors": descriptors}
            self.extracted_img_pts_.append(data)

    def write_keypoints(self, dir_kpts_out : Path):
        assert self.extracted_img_pts_, "No extracted image points found!"
        for data in self.extracted_img_pts_:
            path = dir_kpts_out / f"{str(data["img_path"].name)}.txt"
            lines = [f"{kpt.pt[0]} {kpt.pt[1]} {kpt.size} {kpt.angle}\n" for kpt in data["keypoints"]]
            path.mkdir(exist_ok=True)
            with open(path, "w") as f:
                f.writelines(lines)

    def write_matches(self, dir_matches_out : Path):
        with open(dir_matches_out / "match_list.txt", "w") as f:
            for i, img_pt_i in enumerate(self.extracted_img_pts_):
                if i == len(self.extracted_img_pts_):
                    break
                for j in range(i + 1, len(self.extracted_img_pts_)):
                    img_pt_j = self.extracted_img_pts_[j]
                    f.write(f"{img_pt_i["img_path"]} {img_pt_j["img_path"]}\n")
                    matches = self.bf_matcher_.match(img_pt_i, img_pt_j)
                    with open(dir_matches_out / f"{img_pt_i["img_path"].name}-{img_pt_i["img_path"].name}") as f_j:
                        for match in matches:
                            f_j.write(f"{match.queryIdx} {match.trainIdx}\n")
