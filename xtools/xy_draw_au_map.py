import math
import os
import sys
from os.path import dirname, abspath
from typing import Tuple
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib
matplotlib.use('nbAgg')
import pickle
import re

CONCEPT_MAP = [
        "AU1","AU2","AU4","AU5","AU6","AU7",
        "AU9","AU10","AU12","AU14","AU15","AU17",
        "AU20","AU23","AU25","AU26","AU28","AU45",
    ]

AU = {0: 'Inner Brow Raiser',
      1: 'Outer Brow Raiser',
      2: 'Brow Lowerer',
      3: 'Upper Lid Raiser',
      4: 'Cheek Raiser',
      5: 'Lid Tightener',
      6: 'Nose Wrinkler',
      7: 'Upper Lip Raiser',
      8: 'Lip Corner Puller',
      9: 'Dimpler',
      10: 'Lip Corner Depressor',
      11: 'Chin Raiser',
      12: 'Lip Stretcher',
      13: 'Lip Tightener',
      14: 'Lip pressor',
      15: 'Lips Part',
      16: 'Jaw Drop',
      17: 'Eyes Closed',
      18: 'Mouth Stretcher',
      19: 'Lower Lip Depressor'
      }

EXP2AU = {
    "ANGER": [2, 3, 5, 7, 11, 13, 14, 15, 16],
    "DISGUST": [6, 3, 19, 11, 15, 16],
    "FEAR": [0, 1, 2, 3, 12, 15, 16, 18],
    "HAPPINESS": [4, 8, 15],
    "SADNESS": [0, 2, 4, 10, 11],
    "SURPRISE":[0, 1, 3, 16, 18]
}

GENERIC_AUS = sorted(list(AU.keys()))

ALL_AUS = sorted(list(AU.keys()))

def plot_action_units_ellipsoid(au: int,
                                h: int,
                                w: int,
                                lndmks: list,
                                ) -> Tuple[np.ndarray, bool]:
    assert isinstance(lndmks, list), type(lndmks)
    assert len(lndmks) == 68, len(lndmks)


    att_map = np.zeros((h, w)) + 1e-4
    cp = att_map.copy()
    col = (255, 255, 255)
    a = 0
    s = 0
    e = 360
    f = cv2.FILLED

    if au == 0:
        l_x1, l_y1 = lndmks[20]
        r_x2, r_y2 = lndmks[23]
        major = round(w / 8)
        minor = round(h/10)
        cv2.ellipse(att_map, (l_x1, l_y1), (major, minor), a, s, e, col, f)
        cv2.ellipse(att_map, (r_x2, r_y2), (major, minor), a, s, e, col, f)

    elif au == 1:

        l_x1, l_y1 = lndmks[18]
        r_x2, r_y2 = lndmks[25]
        major = round(w / 8)
        minor = round(h / 10)

        cv2.ellipse(att_map, (l_x1, l_y1), (major, minor), a, s, e, col, f)
        cv2.ellipse(att_map, (r_x2, r_y2), (major, minor), a, s, e, col, f)

    elif au == 2:
        l_x, l_y = lndmks[19]
        r_x, r_y = lndmks[24]
        x = int((l_x + r_x) / 2)
        y = int((l_y + r_y) / 2)
        major = max(int((r_x - l_x) / 2), 0)
        minor = max(int((r_y - l_y) / 2), 0)

        if major == 0:
            major = 10

        if minor == 0:
            minor = 10

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

    elif au == 3:
        l_x1, _ = lndmks[36]
        _, l_y1 = lndmks[38]
        r_x1, _ = lndmks[39]
        _, r_y1 = lndmks[41]
        l_x2, _ = lndmks[42]
        _, l_y2 = lndmks[44]
        r_x2, _ = lndmks[45]
        _, r_y2 = lndmks[47]

        x = int((l_x1 + r_x1) / 2)
        y = int((l_y1 + r_y1) / 2)
        major = max(int((r_x1 - l_x1) / 2), 0)
        minor = max(int((r_y1 - l_y1) / 2), 0)

        if major == 0:
            major = 5

        if minor == 0:
            minor = 5

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

        x = int((l_x2 + r_x2) / 2)
        y = int((l_y2 + r_y2) / 2)
        major = max(int((r_x2 - l_x2) / 2), 0)
        minor = max(int((r_y2 - l_y2) / 2), 0)

        if major == 0:
            major = 5

        if minor == 0:
            minor = 5
        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

    elif au == 4:
        l_x1, l_y1 = lndmks[41]
        r_x1, r_y1 = lndmks[46]
        x = l_x1
        y = l_y1 + round(h / 6)
        major = round(w / 10)
        minor = round(h / 10)
        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)
        x = r_x1
        y = r_y1 + round(h / 6)
        major = round(w / 10)
        minor = round(h / 10)
        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

    elif au == 5:
        l_x1, _ = lndmks[36]
        _, l_y1 = lndmks[38]
        r_x1, _ = lndmks[39]
        _, r_y1 = lndmks[41]
        l_x2, _ = lndmks[42]
        _, l_y2 = lndmks[44]
        r_x2, _ = lndmks[45]
        _, r_y2 = lndmks[47]
        x = int((l_x1 + r_x1) / 2)
        y = int((l_y1 + r_y1) / 2)
        major = max(int((r_x1 - l_x1) / 2), 0)
        minor = max(int((r_y1 + 10 - l_y1 + 10) / 2), 0)
        if major == 0:
            major = 5
        if minor == 0:
            minor = 5
        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)
        x = int((l_x2 + r_x2) / 2)
        y = int((l_y2 + r_y2) / 2)
        major = max(int((r_x2 - l_x2) / 2), 0)
        minor = max(int((r_y2 + 10 - l_y2 + 10) / 2), 0)
        if major == 0:
            major = 5
        if minor == 0:
            minor = 5
        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

    elif au == 6:
        l_x1, l_y1 = lndmks[29]
        r_x1, r_y1 = lndmks[31]
        r_x2, r_y2 = lndmks[35]
        cv2.ellipse(att_map, (r_x1, l_y1), (20, 20), a, s, e, col, f)
        cv2.ellipse(att_map, (r_x2, l_y1), (20, 20), a, s, e, col, f)

    elif au == 7:
        l_x1, l_y1 = lndmks[48]
        l_x2, l_y2 = lndmks[50]
        center_x = (l_x1 + l_x2) // 2
        center_y = (l_y1 + l_y2) // 2
        center = (center_x, center_y)
        major_axis_length = int(math.sqrt((l_x2 - l_x1) ** 2 + (l_y2 - l_y1) ** 2))
        angle = math.degrees(math.atan2(l_y2 - l_y1, l_x2 - l_x1))
        minor_axis_length = major_axis_length // 2
        axes = (major_axis_length // 2, minor_axis_length // 2)
        cv2.ellipse(att_map, center, axes, angle, s, e, col, f)
        r_x1, r_y1 = lndmks[52]
        r_x2, r_y2 = lndmks[54]
        center_x = (r_x1 + r_x2) // 2
        center_y = (r_y1 + r_y2) // 2
        center = (center_x, center_y)
        major_axis_length = int(math.sqrt((l_x2 - l_x1) ** 2 + (l_y2 - l_y1) ** 2))
        angle = math.degrees(math.atan2(l_y2 - l_y1, l_x2 - l_x1))
        minor_axis_length = major_axis_length // 2
        axes = (major_axis_length // 2, minor_axis_length // 2)
        cv2.ellipse(att_map, center, axes, angle, s, e, col, f)

    elif au == 8:
        l_x, l_y = lndmks[48]
        r_x, r_y = lndmks[54]

        major = 20
        minor = 20

        cv2.ellipse(att_map, (l_x, l_y), (major, minor), a, s, e, col, f)
        cv2.ellipse(att_map, (r_x, r_y), (major, minor), a, s, e, col, f)

    elif au == 9:
        l_x, l_y = lndmks[48]
        r_x, r_y = lndmks[54]

        major = 20
        minor = 20

        l_x = max(l_x - 20 , 0)
        r_x = max(min((r_x + 20, w)), min((r_x + 10, w)))

        cv2.ellipse(att_map, (l_x, l_y), (major, minor), a, s, e, col, f)
        cv2.ellipse(att_map, (r_x, r_y), (major, minor), a, s, e, col, f)

    elif au == 10:
        l_x, l_y = lndmks[48]
        r_x, r_y = lndmks[54]

        major = 20
        minor = 20

        cv2.ellipse(att_map, (l_x, l_y), (major, minor), a, s, e, col, f)
        cv2.ellipse(att_map, (r_x, r_y), (major, minor), a, s, e, col, f)

    elif au == 11:
        l_x, l_y = lndmks[59]
        r_x, r_y = lndmks[8]

        x = int((l_x + r_x) / 2)
        y = int((l_y + r_y) / 2)

        major = max(int((r_x - l_x) / 2), 0)
        minor = max(int((r_y - l_y) / 2), 0)

        if major == 0:
            major = 5

        if minor == 0:
            minor = 5

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)


    elif au == 12:
        l_x1, l_y1 = lndmks[48]
        l_x2, l_y2 = lndmks[6]


        r_x1, r_y1 = lndmks[50]
        r_x2, r_y2 = lndmks[10]

        x = int((l_x1 + l_x2) / 2)
        y = int((l_y1 + l_y2) / 2)
        major = 20
        minor = 20

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

        x = int((r_x1 + r_x2) / 2)
        y = int((r_y1 + r_y2) / 2)

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)


    elif au == 13:
        l_x, _ = lndmks[48]
        r_x, _ = lndmks[54]
        _, l_y = lndmks[51]
        _, r_y = lndmks[57]

        x = int((l_x + r_x) / 2)
        y = int((l_y + r_y) / 2)

        major = max(int((r_x - l_x) / 2), 0)
        minor = max(int((r_y - l_y) / 2), 0)

        if major == 0:
            major = 5

        if minor == 0:
            minor = 5

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)


    elif au == 14:
        l_x, _ = lndmks[48]
        r_x, _ = lndmks[54]
        _, l_y = lndmks[51]
        _, r_y = lndmks[57]

        x = int((l_x + r_x) / 2)
        y = int((l_y + r_y) / 2)

        major = max(int((r_x - l_x) / 2), 0)
        minor = max(int((r_y - l_y) / 2), 0)

        if major == 0:
            major = 5

        if minor == 0:
            minor = 5

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

    elif au == 15:

        t_x, t_y = lndmks[51]
        b_x, b_y = lndmks[57]

        major = 25
        minor = 10

        cv2.ellipse(att_map, (t_x, t_y), (major, minor), a, s, e, col, f)
        cv2.ellipse(att_map, (b_x, b_y), (major, minor), a, s, e, col, f)

    elif au == 16:
        l_x, _ = lndmks[48]
        r_x, _ = lndmks[54]
        _, l_y = lndmks[51]
        _, r_y = lndmks[57]

        x = int((l_x + r_x) / 2)
        y = int((l_y + r_y) / 2)
        major = max(int((r_x - l_x) / 2), 0)
        minor = max(int((r_y - l_y) / 2), 0)

        if major == 0:
            major = 5

        if minor == 0:
            minor = 5

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

    elif au == 17:
        l_x1, _ = lndmks[36]
        _, l_y1 = lndmks[38]
        r_x1, _ = lndmks[39]
        _, r_y1 = lndmks[41]
        l_x2, _ = lndmks[42]
        _, l_y2 = lndmks[44]
        r_x2, _ = lndmks[45]
        _, r_y2 = lndmks[47]

        x = int((l_x1 + r_x1) / 2)
        y = int((l_y1 + r_y1) / 2)

        major = max(int((r_x1 - l_x1) / 2), 0)
        minor = max(int((r_y1 + 10 - l_y1 + 10) / 2), 0)

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

        x = int((l_x2 + r_x2) / 2)
        y = int((l_y2 + r_y2) / 2)

        major = max(int((r_x2 - l_x2) / 2), 0)
        minor = max(int((r_y2 + 10 - l_y2 + 10) / 2), 0)
        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

    elif au == 18:
        l_x, _ = lndmks[48]
        r_x, _ = lndmks[54]
        _, l_y = lndmks[51]
        _, r_y = lndmks[57]

        x = int((l_x + r_x) / 2)
        y = int((l_y + r_y) / 2)
        major = max(int((r_x - l_x) / 2), 0)
        minor = max(int((r_y - l_y) / 2), 0)

        if major == 0:
            major = 5

        if minor == 0:
            minor = 5

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

    elif au == 19:
        l_x, l_y = lndmks[59]
        r_x, r_y = lndmks[55]

        major = 20
        minor = 20

        cv2.ellipse(att_map, (l_x, l_y), (major, minor), a, s, e, col, f)
        cv2.ellipse(att_map, (r_x, r_y), (major, minor), a, s, e, col, f)

    is_roi = ((att_map - cp).sum() > 0)


    if is_roi:
        att_map = cv2.resize(att_map, dsize=(28, 28))

    else:
        att_map = np.zeros((28, 28)) + np.inf

    return att_map, is_roi

def plot_action_units_ellipsoid_low(au: int,
                                h: int,
                                w: int,
                                lndmks: list,
                                ) -> Tuple[np.ndarray, bool]:
    assert isinstance(lndmks, list), type(lndmks)
    assert len(lndmks) == 68, len(lndmks)


    att_map = np.zeros((h, w)) + 1e-4
    cp = att_map.copy()
    col = (255, 255, 255)
    a = 0
    s = 0
    e = 360
    f = cv2.FILLED

    if au == 0:
        l_x1, l_y1 = lndmks[20]
        r_x2, r_y2 = lndmks[23]
        major = round(w / 8)
        minor = round(h/10)
        cv2.ellipse(att_map, (l_x1, l_y1), (major, minor), a, s, e, col, f)
        cv2.ellipse(att_map, (r_x2, r_y2), (major, minor), a, s, e, col, f)

    elif au == 1:

        l_x1, l_y1 = lndmks[18]
        r_x2, r_y2 = lndmks[25]
        major = round(w / 8)
        minor = round(h / 10)

        cv2.ellipse(att_map, (l_x1, l_y1), (major, minor), a, s, e, col, f)
        cv2.ellipse(att_map, (r_x2, r_y2), (major, minor), a, s, e, col, f)

    elif au == 2:
        l_x, l_y = lndmks[19]
        r_x, r_y = lndmks[24]
        x = int((l_x + r_x) / 2)
        y = int((l_y + r_y) / 2)
        major = max(int((r_x - l_x) / 2), 0)
        minor = max(int((r_y - l_y) / 2), 0)

        if major == 0:
            major = 10

        if minor == 0:
            minor = 10

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

    elif au == 3:
        l_x1, _ = lndmks[36]
        _, l_y1 = lndmks[38]
        r_x1, _ = lndmks[39]
        _, r_y1 = lndmks[41]
        l_x2, _ = lndmks[42]
        _, l_y2 = lndmks[44]
        r_x2, _ = lndmks[45]
        _, r_y2 = lndmks[47]

        x = int((l_x1 + r_x1) / 2)
        y = int((l_y1 + r_y1) / 2)
        major = max(int((r_x1 - l_x1) / 2), 0)
        minor = max(int((r_y1 - l_y1) / 2), 0)

        if major == 0:
            major = 5

        if minor == 0:
            minor = 5

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

        x = int((l_x2 + r_x2) / 2)
        y = int((l_y2 + r_y2) / 2)
        major = max(int((r_x2 - l_x2) / 2), 0)
        minor = max(int((r_y2 - l_y2) / 2), 0)

        if major == 0:
            major = 5

        if minor == 0:
            minor = 5

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

    elif au == 4:
        l_x1, l_y1 = lndmks[41]
        r_x1, r_y1 = lndmks[46]

        x = l_x1
        y = l_y1 + round(h / 6)
        major = round(w / 10)
        minor = round(h / 10)

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

        x = r_x1
        y = r_y1 + round(h / 6)
        major = round(w / 10)
        minor = round(h / 10)

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

    elif au == 5:
        l_x1, _ = lndmks[36]
        _, l_y1 = lndmks[38]
        r_x1, _ = lndmks[39]
        _, r_y1 = lndmks[41]
        l_x2, _ = lndmks[42]
        _, l_y2 = lndmks[44]
        r_x2, _ = lndmks[45]
        _, r_y2 = lndmks[47]

        x = int((l_x1 + r_x1) / 2)
        y = int((l_y1 + r_y1) / 2)
        major = max(int((r_x1 - l_x1) / 2), 0)
        minor = max(int((r_y1 + 10 - l_y1 + 10) / 2), 0)

        if major == 0:
            major = 5

        if minor == 0:
            minor = 5

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

        x = int((l_x2 + r_x2) / 2)
        y = int((l_y2 + r_y2) / 2)
        major = max(int((r_x2 - l_x2) / 2), 0)
        minor = max(int((r_y2 + 10 - l_y2 + 10) / 2), 0)

        if major == 0:
            major = 5

        if minor == 0:
            minor = 5

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

    elif au == 6:
        l_x1, l_y1 = lndmks[29]
        r_x1, r_y1 = lndmks[31]
        r_x2, r_y2 = lndmks[35]

        cv2.ellipse(att_map, (r_x1, l_y1), (10, 10), a, s, e, col, f)
        cv2.ellipse(att_map, (r_x2, l_y1), (10, 10), a, s, e, col, f)

    elif au == 7:
        l_x1, l_y1 = lndmks[48]
        l_x2, l_y2 = lndmks[50]
        center_x = (l_x1 + l_x2) // 2
        center_y = (l_y1 + l_y2) // 2
        center = (center_x, center_y)

        major_axis_length = int(math.sqrt((l_x2 - l_x1) ** 2 + (l_y2 - l_y1) ** 2))
        angle = math.degrees(math.atan2(l_y2 - l_y1, l_x2 - l_x1))
        minor_axis_length = major_axis_length // 2
        axes = (major_axis_length // 2, minor_axis_length // 2)
        cv2.ellipse(att_map, center, axes, angle, s, e, col, f)

        r_x1, r_y1 = lndmks[52]
        r_x2, r_y2 = lndmks[54]
        center_x = (r_x1 + r_x2) // 2
        center_y = (r_y1 + r_y2) // 2
        center = (center_x, center_y)

        major_axis_length = int(math.sqrt((l_x2 - l_x1) ** 2 + (l_y2 - l_y1) ** 2))
        angle = math.degrees(math.atan2(l_y2 - l_y1, l_x2 - l_x1))
        minor_axis_length = major_axis_length // 2
        axes = (major_axis_length // 2, minor_axis_length // 2)
        cv2.ellipse(att_map, center, axes, angle, s, e, col, f)

    elif au == 8:
        l_x, l_y = lndmks[48]
        r_x, r_y = lndmks[54]

        major = 10
        minor = 10

        cv2.ellipse(att_map, (l_x, l_y), (major, minor), a, s, e, col, f)
        cv2.ellipse(att_map, (r_x, r_y), (major, minor), a, s, e, col, f)

    elif au == 9:
        l_x, l_y = lndmks[48]
        r_x, r_y = lndmks[54]

        major = 10
        minor = 10

        l_x = max(l_x - 10 , 0)
        r_x = max(min((r_x + 10, w)), min((r_x + 5, w)))

        cv2.ellipse(att_map, (l_x, l_y), (major, minor), a, s, e, col, f)
        cv2.ellipse(att_map, (r_x, r_y), (major, minor), a, s, e, col, f)

    elif au == 10:
        l_x, l_y = lndmks[48]
        r_x, r_y = lndmks[54]

        major = 10
        minor = 10

        cv2.ellipse(att_map, (l_x, l_y), (major, minor), a, s, e, col, f)
        cv2.ellipse(att_map, (r_x, r_y), (major, minor), a, s, e, col, f)

    elif au == 11:
        l_x, l_y = lndmks[59]
        r_x, r_y = lndmks[8]
        x = int((l_x + r_x) / 2)
        y = int((l_y + r_y) / 2)
        major = max(int((r_x - l_x) / 2), 0)
        minor = max(int((r_y - l_y) / 2), 0)
        if major == 0:
            major = 4
        if minor == 0:
            minor = 4
        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)


    elif au == 12:
        l_x1, l_y1 = lndmks[48]
        l_x2, l_y2 = lndmks[6]
        r_x1, r_y1 = lndmks[50]
        r_x2, r_y2 = lndmks[10]
        x = int((l_x1 + l_x2) / 2)
        y = int((l_y1 + l_y2) / 2)
        major = 10
        minor = 10
        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)
        x = int((r_x1 + r_x2) / 2)
        y = int((r_y1 + r_y2) / 2)
        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)


    elif au == 13:
        l_x, _ = lndmks[48]
        r_x, _ = lndmks[54]
        _, l_y = lndmks[51]
        _, r_y = lndmks[57]

        x = int((l_x + r_x) / 2)
        y = int((l_y + r_y) / 2)

        major = max(int((r_x - l_x) / 2), 0)
        minor = max(int((r_y - l_y) / 2), 0)

        if major == 0:
            major = 3

        if minor == 0:
            minor = 3

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)


    elif au == 14:
        l_x, _ = lndmks[48]
        r_x, _ = lndmks[54]
        _, l_y = lndmks[51]
        _, r_y = lndmks[57]

        x = int((l_x + r_x) / 2)
        y = int((l_y + r_y) / 2)

        major = max(int((r_x - l_x) / 2), 0)
        minor = max(int((r_y - l_y) / 2), 0)

        if major == 0:
            major = 3

        if minor == 0:
            minor = 3

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

    elif au == 15:
        t_x, t_y = lndmks[51]
        b_x, b_y = lndmks[57]
        major = 20
        minor = 5
        cv2.ellipse(att_map, (t_x, t_y), (major, minor), a, s, e, col, f)
        cv2.ellipse(att_map, (b_x, b_y), (major, minor), a, s, e, col, f)

    elif au == 16:
        l_x, _ = lndmks[48]
        r_x, _ = lndmks[54]
        _, l_y = lndmks[51]
        _, r_y = lndmks[57]

        x = int((l_x + r_x) / 2)
        y = int((l_y + r_y) / 2)
        major = max(int((r_x - l_x) / 2), 0)
        minor = max(int((r_y - l_y) / 2), 0)

        if major == 0:
            major = 5

        if minor == 0:
            minor = 5

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

    elif au == 17:
        l_x1, _ = lndmks[36]
        _, l_y1 = lndmks[38]
        r_x1, _ = lndmks[39]
        _, r_y1 = lndmks[41]
        l_x2, _ = lndmks[42]
        _, l_y2 = lndmks[44]
        r_x2, _ = lndmks[45]
        _, r_y2 = lndmks[47]

        x = int((l_x1 + r_x1) / 2)
        y = int((l_y1 + r_y1) / 2)

        major = max(int((r_x1 - l_x1) / 2), 0)
        minor = max(int((r_y1 + 5 - l_y1 + 5) / 2), 0)

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

        x = int((l_x2 + r_x2) / 2)
        y = int((l_y2 + r_y2) / 2)

        major = max(int((r_x2 - l_x2) / 2), 0)
        minor = max(int((r_y2 + 5 - l_y2 + 5) / 2), 0)
        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

    elif au == 18:
        l_x, _ = lndmks[48]
        r_x, _ = lndmks[54]
        _, l_y = lndmks[51]
        _, r_y = lndmks[57]

        x = int((l_x + r_x) / 2)
        y = int((l_y + r_y) / 2)
        major = max(int((r_x - l_x) / 2), 0)
        minor = max(int((r_y - l_y) / 2), 0)

        if major == 0:
            major = 5

        if minor == 0:
            minor = 5

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

    elif au == 19:
        l_x, l_y = lndmks[59]
        r_x, r_y = lndmks[55]
        major = 10
        minor = 10
        cv2.ellipse(att_map, (l_x, l_y), (major, minor), a, s, e, col, f)
        cv2.ellipse(att_map, (r_x, r_y), (major, minor), a, s, e, col, f)
    is_roi = ((att_map - cp).sum() > 0)


    if is_roi:
        att_map = cv2.resize(att_map, dsize=(28, 28))
    else:
        att_map = np.zeros((28, 28)) + np.inf

    return att_map, is_roi

def build_all_action_units(lndmks: list,
                           h: int,
                           w: int,
                           low: bool = False
                           ) -> np.ndarray:

    assert isinstance(lndmks, list), type(lndmks)
    assert len(lndmks) == 68, len(lndmks)
    lndmks = [(int(z[0]), int(z[1])) for z in lndmks]
    aus = ALL_AUS
    n_au = 20
    full_att_map = np.zeros((n_au, h, w))
    if lndmks[0][0] == np.inf:
        return np.zeros((1, h, w)) + np.inf
    rois = []
    holder = []
    failed = 0
    for i, au in enumerate(aus):
        if low:
            att_map, is_roi = plot_action_units_ellipsoid_low(au=au, h=h, w=w, lndmks=lndmks)
        else:
            att_map, is_roi = plot_action_units_ellipsoid(
            au=au, h=h, w=w, lndmks=lndmks)
        if is_roi:
            att_map = cv2.blur(att_map, (3, 3))
            attmap_resized = cv2.resize(att_map, dsize=(w, h))
            full_att_map[i, :, :] = attmap_resized
            holder.append(attmap_resized)
        else:
            failed += 1
        rois.append(is_roi)
    if sum(rois) == 0:
        return np.zeros((1, h, w)) + np.inf
    if failed > 0:
        full_att_map = np.array(holder)
    return full_att_map.astype(np.float32)


def fast_draw_heatmap(img: np.ndarray,
                      heatmaps: np.ndarray,
                      wfp: str,
                      normalize: bool = True,
                      binary_roi: np.ndarray = None,
                      img_msk_black: np.ndarray = None,
                      img_msk_avg: np.ndarray = None,
                      img_msk_blur: np.ndarray = None):
    if normalize:
        heatmaps_min = heatmaps.min(axis=(1, 2), keepdims=True)
        heatmaps_max = heatmaps.max(axis=(1, 2), keepdims=True)
        heatmaps_range = heatmaps_max - heatmaps_min
        heatmaps_range[heatmaps_range == 0] = 1
        heatmaps = (heatmaps - heatmaps_min) / heatmaps_range
    n_au = heatmaps.shape[0]
    ncols = n_au
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(3 * ncols, 3), squeeze=False)
    for i in range(ncols):
        axes[0, i].imshow(img[:, :, ::-1])
        axes[0, i].imshow(heatmaps[i], alpha=0.7, cmap='jet')
        axes[0, i].text(
            3, 40, f'AU {i}',
            fontsize=7,
            bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8, 'edgecolor': 'none'}
        )
        axes[0, i].axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig(wfp, pad_inches=0, bbox_inches='tight', dpi=300)
    plt.close()


def convert_to_patch_level_heatmap(au_maps: np.ndarray, patch_size: Tuple[int, int] = (14, 14)) -> np.ndarray:
    n_au, img_h, img_w = au_maps.shape
    patch_h, patch_w = patch_size
    patch_level_maps = np.zeros((n_au, patch_h, patch_w))
    for i in range(n_au):
        patch_level_maps[i] = cv2.resize(au_maps[i], (patch_w, patch_h), interpolation=cv2.INTER_AREA)
    return patch_level_maps

def visualize_patch_based_heatmap_old(img: np.ndarray,
                                  heatmaps: np.ndarray,
                                  save_path: str,
                                  normalize: bool = True,
                                  label: bool = True):
    if normalize:
        heatmaps_min = heatmaps.min(axis=(1, 2), keepdims=True)
        heatmaps_max = heatmaps.max(axis=(1, 2), keepdims=True)
        heatmaps_range = heatmaps_max - heatmaps_min
        heatmaps_range[heatmaps_range == 0] = 1
        heatmaps = (heatmaps - heatmaps_min) / heatmaps_range
    n_au = heatmaps.shape[0]
    patch_h, patch_w = heatmaps.shape[1:3]
    img_h, img_w = img.shape[:2]
    fig, axes = plt.subplots(nrows=1, ncols=n_au, figsize=(3 * n_au, 3), squeeze=False)

    for i in range(n_au):
        heatmap_resized = cv2.resize(heatmaps[i], (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        axes[0, i].imshow(img[:, :, ::-1])
        axes[0, i].imshow(heatmap_resized, alpha=0.5, cmap='magma')
        if label:
            axes[0, i].text(
                3, 40, f'C_{i}',
                fontsize=8,
                color='black',
                bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8, 'edgecolor': 'none'}
            )

        axes[0, i].axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    if save_path != '':
        plt.savefig(save_path.replace('bmp','jpg'), pad_inches=0, bbox_inches='tight', dpi=300, transparent=True)
    plt.close()

def visualize_patch_based_heatmap(img: np.ndarray,
                                  heatmaps: np.ndarray,
                                  save_path: str,
                                  normalize: bool = True,
                                  label: bool = True,
                                  separate: bool = False):
    assert len(CONCEPT_MAP) == heatmaps.shape[0], \
        f"Number of heatmaps ({heatmaps.shape[0]}) does not match CONCEPT_MAP ({len(CONCEPT_MAP)})"

    if normalize:
        heatmaps_min = heatmaps.min(axis=(1, 2), keepdims=True)
        heatmaps_max = heatmaps.max(axis=(1, 2), keepdims=True)
        heatmaps_range = heatmaps_max - heatmaps_min
        heatmaps_range[heatmaps_range == 0] = 1
        heatmaps = (heatmaps - heatmaps_min) / heatmaps_range

    n_au = heatmaps.shape[0]
    img_h, img_w = img.shape[:2]
    saved_paths = []
    if separate:
        base_dir = os.path.dirname(save_path)
        os.makedirs(base_dir, exist_ok=True)
        for i, au_name in enumerate(CONCEPT_MAP[:n_au]):
            heatmap_resized = cv2.resize(heatmaps[i], (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(img[:, :, ::-1])
            ax.imshow(heatmap_resized, alpha=0.5, cmap='magma')
            ax.axis('off')
            if label:
                ax.text(
                    3, 40, au_name,
                    fontsize=8,
                    color='black',
                    bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8, 'edgecolor': 'none'}
                )
            clean_name = au_name.replace(":", "_")
            single_save_path = save_path.replace('.bmp', f'_{clean_name}.jpg').replace('.jpg', f'_{clean_name}.jpg')
            plt.savefig(single_save_path, pad_inches=0, bbox_inches='tight', dpi=300, transparent=True)
            plt.close(fig)
            saved_paths.append(single_save_path)
        return saved_paths
    fig, axes = plt.subplots(nrows=1, ncols=n_au, figsize=(3 * n_au, 3), squeeze=False)
    for i, au_name in enumerate(CONCEPT_MAP[:n_au]):
        heatmap_resized = cv2.resize(heatmaps[i], (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        axes[0, i].imshow(img[:, :, ::-1])
        axes[0, i].imshow(heatmap_resized, alpha=0.5, cmap='magma')
        if label:
            axes[0, i].text(
                3, 40, au_name,
                fontsize=8,
                color='black',
                bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8, 'edgecolor': 'none'}
            )
        axes[0, i].axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    if save_path:
        plt.savefig(save_path.replace('bmp', 'jpg'), pad_inches=0, bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)

    return [save_path]
    fig, axes = plt.subplots(nrows=1, ncols=n_au, figsize=(3 * n_au, 3), squeeze=False)
    for i in range(n_au):
        heatmap_resized = cv2.resize(heatmaps[i], (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        axes[0, i].imshow(img[:, :, ::-1])
        axes[0, i].imshow(heatmap_resized, alpha=0.5, cmap='magma')

        if label:
            axes[0, i].text(
                3, 40, f'C_{i}',
                fontsize=8,
                color='black',
                bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8, 'edgecolor': 'none'}
            )
        axes[0, i].axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    if save_path:
        plt.savefig(save_path.replace('bmp', 'jpg'), pad_inches=0, bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)

def show_GT_au_heatmap_for_image(image_path, save_path, save = True, dark=False, label=True):
    landmarks_file = image_path.replace('images', 'annotations').replace('.jpg', '_lnd.npy')
    au_path = image_path.replace('images', 'AUs').replace('.jpg', '.csv')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    output_dir = os.path.dirname(save_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(au_path):
        print(f"File {au_path} does not exist.")
        return
    au_ = pd.read_csv(au_path)
    au_ = au_.values.tolist()[0]
    au_label = au_[19:39]
    au_label = [int(x) for x in au_label]
    image = cv2.imread(image_path)
    if dark:
        image = np.zeros_like(image)
    lndmks = np.load(landmarks_file)

    if lndmks.shape == (136,):
        lndmks = lndmks.reshape(68, 2)
    elif lndmks.shape != (68, 2):
        raise ValueError(f"Expected landmarks to be of shape (68, 2), but got {lndmks.shape}")
    lndmks = [tuple(point) for point in lndmks]
    h, w = image.shape[:2]
    au_maps = build_all_action_units(lndmks, h, w)
    au_maps_patch_level = convert_to_patch_level_heatmap(au_maps[:18,:,:], patch_size=(14, 14))
    if save:
        visualize_patch_based_heatmap(image, au_maps_patch_level, save_path, normalize=True, label=label)
    return au_maps_patch_level


def visualize_weighted_heatmaps(img: np.ndarray,
                                heatmaps: np.ndarray,
                                weights: np.ndarray,
                                save_path: str = '',
                                normalize: bool = True,
                                show: bool = True):
    assert heatmaps.shape[0] == weights.shape[0], "Number of heatmaps and weights must match."

    if np.sum(weights) == 0:
        img_h, img_w = img.shape[:2]
        weighted_heatmap = np.zeros((img_h, img_w), dtype=np.float32)
        if normalize:
            weighted_heatmap = (weighted_heatmap - weighted_heatmap.min()) / (
                    weighted_heatmap.max() - weighted_heatmap.min() + 1e-10)
        plt.figure(figsize=(6, 6))
        plt.imshow(img[:, :, ::-1])
        plt.imshow(weighted_heatmap, alpha=0.5, cmap='magma')
        plt.axis('off')
        if save_path != '':
            plt.savefig(save_path, pad_inches=0, bbox_inches='tight', dpi=300)
        if show:
            plt.show()
        else:
            plt.close()
        return
    weights = weights / weights.sum()
    n_heatmaps = heatmaps.shape[0]
    img_h, img_w = img.shape[:2]
    weighted_heatmap = np.zeros((img_h, img_w), dtype=np.float32)
    for i in range(n_heatmaps):
        heatmap_resized = cv2.resize(heatmaps[i], (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        if normalize:
            heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (
                        heatmap_resized.max() - heatmap_resized.min() + 1e-10)
        weighted_heatmap += weights[i] * heatmap_resized
    if normalize:
        weighted_heatmap = (weighted_heatmap - weighted_heatmap.min()) / (
                    weighted_heatmap.max() - weighted_heatmap.min() + 1e-10)
    plt.figure(figsize=(6, 6))
    plt.imshow(img[:, :, ::-1])
    plt.imshow(weighted_heatmap, alpha=0.5, cmap='magma')
    plt.axis('off')
    if save_path != '':
        plt.savefig(save_path.replace('bmp','jpg'), pad_inches=0, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    else:
        plt.close()

def visualize_GT_weighted_heatmaps(image_path, save_path, au_maps_patch_level):
    image = cv2.imread(image_path)
    au_path = image_path.replace('images', 'AUs').replace('.jpg', '.csv')
    au_ = pd.read_csv(au_path)
    au_ = au_.values.tolist()[0]
    au_label = np.array(au_[19:39])
    visualize_weighted_heatmaps(image, au_maps_patch_level, au_label, save_path, normalize=True)