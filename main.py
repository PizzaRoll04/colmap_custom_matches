from extraction import Extractor
import argparse
from pathlib import Path
from enum import IntEnum
import numpy as np

class CalibIdx(IntEnum):
    FX = 0
    FY = 1
    CX = 2 
    CY = 3
    K1 = 4
    K2 = 5
    K3 = 6
    K4 = 7

def get_k_d(path_calibration:Path):
    with open(path_calibration, "r") as f:
        parsed_cal = f.readline().strip().split(" ")
        values = list(map(float, parsed_cal[1:]))
        K = np.array([[values[CalibIdx.FX], 0, values[CalibIdx.CX]], [0, values[CalibIdx.FY], values[CalibIdx.CY]], [0,0,1]])
        D = np.array([values[CalibIdx.K1], values[CalibIdx.K2], values[CalibIdx.K3], values[CalibIdx.K4]])
        return K, D

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some images.")
    parser.add_argument("--input", type=str, required=True, help="Path to input image directory")
    parser.add_argument("--calibration", type=str, required=True, help="Path to calibration text file")
    parser.add_argument("--build_dir", type=str, required=True, help="Path to create output")

    args = parser.parse_args()
    build_dir = Path(args.build_dir)
    input_dir = Path(args.input)

    extractor = Extractor()
    K, D = get_k_d(args.calibration)
    extractor.extract_images(input_dir, K, D)
    print("extracted images!")
    extractor.write_keypoints(build_dir / "kpts")
    print("keypoints written!")
    extractor.write_matches(build_dir / "matches")
    print("matches written!")

    print("colmap matching extraction complete!")