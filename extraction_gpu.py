from pathlib import Path
from typing import TypedDict
import cv2
import numpy as np

class ExtractedImagePt(TypedDict):
    img_path: Path
    keypoints: list  # CPU-side cv2.KeyPoint list
    descriptors: np.ndarray  # CPU-side descriptors

class ExtractorGPU:
    image_extensions_ = (".jpg", ".jpeg", ".png")

    def __init__(self):
        # Use CUDA SIFT
        self.extractor_ = cv2.cuda.SIFT_create()
        self.bf_matcher_ = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_L2)
        self.extracted_img_pts_: list[ExtractedImagePt] = []

    def extract_images(self, directory: Path, K: np.ndarray = None, D: np.ndarray = None):
        image_files = [f for f in directory.iterdir()
                       if f.is_file() and f.suffix.lower() in self.image_extensions_]
        image_files.sort()

        for image_file in image_files:
            image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Warning: couldn't read {image_file}")
                continue

            if K is not None and D is not None:
                h, w = image.shape[:2]
                image = cv2.fisheye.undistortImage(image, K, D, Knew=K, new_size=(w, h))

            # Upload to GPU
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(image)

            # GPU SIFT extraction
            keypoints_gpu, descriptors_gpu = self.extractor_.detectAndComputeAsync(gpu_img, None)

            if descriptors_gpu is None:
                print(f"Warning: no descriptors found in {image_file}")
                continue

            # Download to CPU
            keypoints = self.extractor_.convert(keypoints_gpu)
            descriptors = descriptors_gpu.download()

            data: ExtractedImagePt = {
                "img_path": image_file,
                "keypoints": keypoints,
                "descriptors": descriptors
            }
            self.extracted_img_pts_.append(data)

    def write_keypoints(self, dir_kpts_out: Path):
        assert self.extracted_img_pts_, "No extracted image points found!"
        for data in self.extracted_img_pts_:
            path = dir_kpts_out / f"{data['img_path'].name}.txt"
            lines = [f"{kpt.pt[0]} {kpt.pt[1]} {kpt.size} {kpt.angle}\n" for kpt in data["keypoints"]]
            path.parent.mkdir(exist_ok=True, parents=True)
            with open(path, "w") as f:
                f.writelines(lines)

    def write_matches(self, dir_matches_out: Path, min_matches: int = None):
        dir_matches_out.mkdir(parents=True, exist_ok=True)
        with open(dir_matches_out / "match_list.txt", "w") as f:
            for i, img_pt_i in enumerate(self.extracted_img_pts_):
                for j in range(i + 1, len(self.extracted_img_pts_)):
                    img_pt_j = self.extracted_img_pts_[j]

                    # Upload descriptors to GPU
                    desc1_gpu = cv2.cuda_GpuMat()
                    desc2_gpu = cv2.cuda_GpuMat()
                    desc1_gpu.upload(img_pt_i["descriptors"])
                    desc2_gpu.upload(img_pt_j["descriptors"])

                    matches_gpu = self.bf_matcher_.match(desc1_gpu, desc2_gpu)
                    matches = sorted(matches_gpu, key=lambda m: m.distance)

                    if min_matches is not None and len(matches) < min_matches:
                        continue

                    f.write(f"{img_pt_i['img_path'].name} {img_pt_j['img_path'].name}\n")
                    match_path = dir_matches_out / f"{img_pt_i['img_path'].name}-{img_pt_j['img_path'].name}.txt"
                    with open(match_path, "w") as f_j:
                        for match in matches:
                            f_j.write(f"{match.queryIdx} {match.trainIdx}\n")
