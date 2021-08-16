import os
import numpy as np

SK_KWARGS = dict(
    invalidation_d=8000,
    collapse_function="sphere",
    soma_radius=8000,
    compute_radius=False,
    shape_function="single",
)

CONTRAST_LOOKUP = {
    "minnie65_phase3_v1": {"black": 0.35, "white": 0.7},
    "v1dd": {"black": 0.35, "white": 0.7},
}

EP_PROOFREADING_TAGS = ["checked", "error", "correct"]
BP_PROOFREADING_TAGS = ["checked", "error"]

res = os.environ.get("GUIDEBOOK_EXPECTED_RESOLUTION", "4,4,40")
GUIDEBOOK_EXPECTED_RESOLUTION = np.array([r for r in map(int, res.split(","))])

SHORT_SEGMENT_THRESH = 5000