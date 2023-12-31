# -*- coding: utf-8 -*-
"""
@author: Rato-12
"""

import numpy as np
import pandas as pd


class dds():
    data = pd.DataFrame(
        index=[0, 1, 2, 3, 4],
        columns=[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        )

    data[4][0] = np.array([
        0, 1, -1, -1, -1, 0, -1, 1,
        -1, -1, 0, -1, -1, 1, 1, 0])

    data[5][0] = np.array([
        0, 1, 1, -1, -1, 1, 0, -1, -1, 1, 1, -1,
        0, 1, -1, 1, -1, 1, 0, 1, 1, 1, 1, 1, 0])

    data[6][0] = np.array([
        0, 1, -1, -1, -1, -1, 1, 0, -1, 1, 1, -1, -1, -1, 0, 1, -1, -1,
        -1, 1, 1, 0, 1, -1, 1, -1, 1, -1, 0, -1, 1, 1, 1, 1, -1, 0])

    data[7][0] = np.array([
        0, 1, -1, 1, -1, 1, -1,
        -1, 0, 1, -1, 1, 1, -1,
        1, -1, 0, 1, 1, 1, 1,
        1, -1, -1, 0, 1, -1, -1,
        -1, -1, 1, 1, 0, -1, -1,
        -1, 1, -1, 1, 1, 0, 1,
        1, 1, 1, 1, 1, -1, 0])

    data[8][0] = np.array([
        0, -1, 1, 1, -1, 1, 1, 1, -1, 0, -1, 1, 1, 1, 1, -1,
        -1, -1, 0, 1, 1, -1, -1, 1, 1, -1, 1, 0, 1, 1, -1, -1,
        -1, -1, 1, -1, 0, -1, 1, -1, 1, -1, -1, -1, 1, 0, 1, 1,
        -1, 1, 1, -1, 1, 1, 0, 1, 1, 1, 1, 1, 1, -1, 1, 0])
    data[9][0] = np.array([
        0, 1, 1, 1, 1, 1, 1, 1, 1,  1, 0, 1, -1, 1, -1, -1, 1, -1,
        -1, 1, 0, -1, 1, -1, 1, -1, -1,  -1, -1, 1, 0, 1, -1, -1, -1, 1,
        1, -1, 1, -1, 0, 1, 1, -1, -1, -1, -1, -1, -1, 1, 0, 1, 1, 1,
        1, 1, -1, -1, 1, 1, 0, -1, 1, -1, -1, -1, 1, 1, 1, -1, 0, -1,
        -1, 1, 1, -1, -1, 1, -1, 1, 0])
    data[10][0] = np.array([
        0, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 0, -1, 1, 1, -1, 1, 1, -1, -1,
        -1, 1, 0, -1, -1, -1, 1, -1, -1, -1,  -1, 1, 1, 0, 1, -1, -1, 1, 1, -1,
        -1, -1, -1, -1, 0, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 0, 1, -1, 1, 1,
        1, 1, -1, -1, -1, -1, 0, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 0, 1, -1,
        1, 1, -1, -1, 1, 1, -1, -1, 0, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 0])
    data[11][0] = np.array([
        0, -1, 1, -1, -1, -1, -1, -1, 1, -1,
        1, -1, 0, -1, -1, 1, -1, -1, -1, -1, 1, 1,
        -1, -1, 0, 1, 1, 1, 1, -1, -1, -1, 1,
        -1, -1, -1, 0, -1, 1, 1, -1, 1, 1, -1,
        1, -1, -1, 1, 0, 1, -1, 1, 1, 1, 1, -1,
        -1, 1, 1, -1, 0, -1, 1, -1, 1, -1,
        -1, -1, -1, 1, 1, -1, 0, 1, 1, -1, -1, -1,
        1, 1, 1, -1, -1, 1, 0, 1, 1, 1,
        -1, 1, -1, -1, -1, 1, -1, 1, 0, -1, 1, 1, -1,
        -1, -1, -1, -1, 1, 1, -1, 0, 1,
        1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 0])
    data[12][0] = np.array([
        0, -1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1,
        -1, 0, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1,
        1, 1, 0, -1, 1, 1, -1, 1, 1, -1, 1, 1,
        1, -1, -1, 0, 1, -1, 1, -1, 1, -1, -1, 1,
        1, 1, 1, 1, 0, -1, 1, 1, 1, 1, 1, 1,
        1, -1, 1, -1, 1, 0, 1, 1, -1, 1, -1, 1,
        1, 1, 1, 1, -1, 1, 0, -1, -1, -1, -1, 1,
        -1, -1, 1, 1, 1, -1, -1, 0, -1, -1, 1, 1,
        1, -1, 1, 1, 1, 1, -1, -1, 0, 1, 1, -1,
        1, 1, -1, 1, 1, -1, -1, 1, -1, 0, -1, -1,
        -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 0, 1,
        1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, 0])

    data[4][1] = np.array([
        0, 1, 1, 1, 1, 1, 0, -1, 1, 1, 1, -1, 0, -1, 1, 1, 1, -1, 0, -1,
        1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 0, 0, 0, 0, -1])
    data[5][1] = np.array([
        0, 1, 1, 1, 1, 1, 1, 0, -1, 1, 1, -1,
        1, -1, 0, -1, 1, 1, 1, 1, -1, 0, -1, 1,
        1, 1, 1, -1, 0, -1, 1, -1, 1, 1, -1, 1,
        0, 0, 0, 0, 0, -1])
    data[6][1] = np.array([
        0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, -1, 1, -1,
        1, -1, 0, 1, 1, -1, 1, 1, -1, -1, 0, 1, 1, -1,
        1, 1, -1, -1, 0, 1, 1, 1, -1, 1, -1, -1, 0, 1,
        1, 1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1,
        0, 0, 0, 0, 0, 0, -1])
    data[7][1] = np.array([
        0, 1, 1, 1, 1, 1, 1, 1,  1, 0, 1, 1, -1, 1, -1, -1,
        1, -1, 0, 1, 1, -1, 1, -1, 1, 1, -1, 0, 1, 1, -1, 1,
        1, 1, -1, -1, 0, 1, 1, -1, 1, -1, 1, -1, -1, 0, 1, 1,
        1, 1, -1, 1, -1, -1, 0, 1, 1, 1, 1, -1, 1, -1, -1, 1,
        0, 0, 0, 0, 0, 0, 0, -1])
    data[8][1] = np.array([
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, -1, -1, -1, -1, 1, 1, 1,
        1, -1, 0, -1, 1, 1, -1, -1, 1, 1, -1, -1, 0, 1, 1, 1, 1, -1,
        1, -1, 1, 1, 0, -1, -1, 1, -1, 1, -1, 1, 1, -1, 0, 1, -1, 1,
        1, 1, -1, 1, -1, 1, 0, -1, -1, 1, 1, -1, 1, 1, -1, -1, 0, 1,
        1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1,
        0, 0, 0, 0, 0, 0, 0, 0, -1])
    data[9][1] = np.array([
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, -1, -1, -1, -1, 1, 1, 1, 1,
        1, -1, 0, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 0, 1, 1, 1, 1, -1, -1,
        1, -1, 1, 1, 0, -1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 0, 1, -1, 1, -1,
        1, 1, -1, 1, -1, 1, 0, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 0, 1, -1,
        1, 1, 1, -1, -1, 1, -1, 1, 0, -1, 1, 1, 1, -1, 1, -1, 1, -1, -1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, -1])
    data[10][1] = np.array([
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, -1, 1, 1, 1, -1, -1,
        -1, 1, 1, -1, 0, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 0, 1, -1,
        1, 1, 1, -1, -1, 1, -1, 1, -1, 0, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1,
        -1, 0, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 0, 1, -1, 1, 1, 1, 1, -1,
        -1, -1, 1, -1, 0, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 0, 1, -1, 1, 1,
        1, 1, -1, -1, -1, 1, -1, 0, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1,
        1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1])
    data[11][1] = np.array([
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, -1, 1, 1, 1, -1, -1, -1,
        1, -1, 1, -1, 0, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 0, 1, -1, 1,
        1, 1, -1, -1, -1, 1, -1, 1, -1, 0, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1,
        1, -1, 0, 1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 0, 1, -1, 1, 1, 1,
        1, 1, -1, -1, -1, 1, -1, 0, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 0,
        1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 0, 1, -1, 1, -1, 1, 1, 1, -1,
        -1, -1, 1, -1, 0, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1])
    data[12][1] = np.array([
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1,
        1, -1, 0, -1, 1, -1, -1, 1, 1, 1, 1, -1,
        -1, 1, 1, -1, 0, -1, 1, -1, -1, 1, 1, 1, 1, -1,
        1, -1, 1, -1, 0, -1, 1, -1, -1, 1, 1, 1, 1, 1,
        -1, -1, 1, -1, 0, -1, 1, -1, -1, 1, 1, 1,
        1, 1, -1, -1, 1, -1, 0, -1, 1, -1,
        -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, 0, -1, 1, -1, -1, 1,
        1, 1, 1, 1, -1, -1, 1, -1, 0, -1, 1, -1,
        -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, 0, -1, 1, -1,
        1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 0, -1, 1, 1,
        -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 0, -1,
        1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1,
        -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1])

    data[4][2] = np.array([
        0, 1, 1, 1, 1, 1, 1, 0, -1, 1, 1, -1, 1, -1, 0, -1, 1, 1,
        1, 1, -1, 0, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1,
        0, 0, 0, 0, -1, -1])
    data[5][2] = np.array([
        0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, -1, 1, -1,
        1, -1, 0, 1, 1, -1, 1, 1, -1, -1, 0, 1, 1, -1,
        1, 1, -1, -1, 0, 1, 1, 1, -1, 1, -1, -1, 1, 1,
        1, 1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1,
        0, 0, 0, 0, 0, -1, -1])
    data[6][2] = np.array([
        0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, -1, 1, -1, -1,
        1, -1, 0, 1, 1, -1, 1, -1, 1, -1, -1, 0, 1, 1, -1, 1,
        1, 1, -1, -1, 0, 1, 1, -1, 1, -1, 1, -1, -1, 0, 1, 1,
        1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, 1, -1, -1, 1,
        0, 0, 0, 0, 0, 0, -1, -1])
    data[7][2] = np.array([
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, -1, -1, -1, -1, 1, 1, 1,
        1, -1, 0, -1, 1, 1, -1, -1, 1, 1, -1, -1, 0, 1, 1, 1, 1, -1,
        1, -1, 1, 1, 0, -1, -1, 1, -1, 1, -1, 1, 1, -1, 0, 1, -1, 1,
        1, 1, -1, 1, -1, 1, 0, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, 1,
        1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1,
        0, 0, 0, 0, 0, 0, 0, -1, -1])
    data[8][2] = np.array([
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, -1, -1, -1, -1, 1, 1, 1, 1,
        1, -1, 0, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 0, 1, 1, 1, 1, -1, -1,
        1, -1, 1, 1, 0, -1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 0, 1, -1, 1, -1,
        1, 1, -1, 1, -1, 1, 0, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 0, 1, -1,
        1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, -1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, -1, -1])
    data[9][2] = np.array([
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, -1, 1, 1, 1, -1, -1,
        -1, 1, 1, -1, 0, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 0, 1,
        -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 0, 1, -1, 1, 1, 1, -1, 1,
        -1, -1, 1, -1, 0, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 0, 1,
        -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 0, 1, -1, 1, 1, 1, 1, -1,
        -1, -1, 1, -1, 0, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1,
        1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1,
        -1, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1])
    data[10][2] = np.array([
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, -1, 1, 1, 1, -1, -1,
        -1, 1, -1, 1, -1, 0, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 0,
        1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 1, -1, 0, 1, -1, 1, 1, 1, -1,
        -1, 1, -1, -1, 1, -1, 0, 1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1,
        0, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 0, 1, -1, 1, 1, 1, 1,
        1, -1, -1, -1, 1, -1, 0, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1,
        0, 1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1,
        1, -1, -1, -1, 1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1])
    data[11][2] = np.array([
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1,
        1, -1, 0, -1, 1, -1, -1, 1, 1, 1, 1, -1,
        -1, 1, 1, -1, 0, -1, 1, -1, -1, 1, 1, 1, 1, -1,
        1, -1, 1, -1, 0, -1, 1, -1, -1, 1, 1, 1, 1, 1,
        -1, -1, 1, -1, 0, -1, 1, -1, -1, 1, 1, 1,
        1, 1, -1, -1, 1, -1, 0, -1, 1, -1,
        -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, 0, -1, 1, -1, -1, 1,
        1, 1, 1, 1, -1, -1, 1, -1, 0, -1, 1, -1,
        -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, 0, -1, 1, -1,
        1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 0, -1, 1, 1,
        -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, -1,
        1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1,
        -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1])
    data[12][2] = np.array([
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 0, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1,
        1, -1, 0, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1,
        1, 1, -1, 0, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1,
        1, -1, 1, -1, 0, -1, 1, -1, -1, 1, 1, 1, 1, -1,
        1, -1, -1, 1, -1, 0, -1, 1, -1, -1, 1, 1, 1, 1,
        1, 1, -1, -1, 1, -1, 0, -1, 1, -1, -1, 1, 1, 1,
        1, 1, 1, -1, -1, 1, -1, 0, -1, 1, -1, -1, 1, 1,
        1, 1, 1, 1, -1, -1, 1, -1, 0, -1, 1, -1, -1, 1,
        1, 1, 1, 1, 1, -1, -1, 1, -1, 0, -1, 1, -1, -1,
        1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 0, -1, 1, -1,
        1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 0, -1, 1,
        1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, -1,
        1, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1])

    data[4][3] = np.array([
        0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, -1, 1, -1,
        1, -1, 0, 1, 1, -1, 1, 1, -1, -1, 0, 1, 1, -1,
        1, 1, -1, -1, 1, 1,  1, -1, 1, -1, -1, 1, 1, 1,
        1, 1, -1, 1, -1, -1, 1,  1, 1, 1, -1, 1, -1, -1,
        0, 0, 0, 0, -1, -1, -1])
    data[5][3] = np.array([
        0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, -1, 1, -1, -1,
        1, -1, 0, 1, 1, -1, 1, -1, 1, -1, -1, 0, 1, 1, -1, 1,
        1, 1, -1, -1, 0, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, 1,
        1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, 1, -1, -1, 1,
        0, 0, 0, 0, 0, -1, -1, -1])
    data[6][3] = np.array([
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, -1, -1, -1, -1, 1, 1, 1,
        1, -1, 0, -1, 1, 1, -1, -1, 1, 1, -1, -1, 0, 1, 1, 1, 1, -1,
        1, -1, 1, 1, 0, -1, -1, 1, -1, 1, -1, 1, 1, -1, 0, 1, -1, 1,
        1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, 1,
        1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1,
        0, 0, 0, 0, 0, 0, -1, -1, -1])
    data[7][3] = np.array([
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, -1, -1, -1, -1, 1, 1, 1, 1,
        1, -1, 0, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 0, 1, 1, 1, 1, -1, -1,
        1, -1, 1, 1, 0, -1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 0, 1, -1, 1, -1,
        1, 1, -1, 1, -1, 1, 0, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1,
        1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, -1, 1,
        0, 0, 0, 0, 0, 0, 0, -1, -1, -1])
    data[8][3] = np.array([
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 0, 1, -1, 1, 1, 1, -1, -1, -1,
        1, 1, -1, 0, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 0, 1, -1, 1, 1, 1,
        -1, -1, 1, -1, 1, -1, 0, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1, -1, 0, 1,
        -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 0, 1, -1, 1, 1,  1, 1, -1, -1, -1,
        1, -1, 0, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 1, 1, 1,
        -1, -1, -1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1,
        -1, 1, 1, 1, -1, -1, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1])
    data[9][3] = np.array([
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, -1, 1, 1, 1, -1, -1, -1,
        1, -1, 1, -1, 0, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 0, 1, -1, 1,
        1, 1, -1, -1, -1, 1, -1, 1, -1, 0, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1,
        1, -1, 0, 1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 0, 1, -1, 1, 1, 1,
        1, 1, -1, -1, -1, 1, -1, 0, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 0,
        1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1,
        -1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1])
    data[10][3] = np.array([
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1,
        1, -1, 0, -1, 1, -1, -1, 1, 1, 1, 1, -1,
        -1, 1, 1, -1, 0, -1, 1, -1, -1, 1, 1, 1, 1, -1,
        1, -1, 1, -1, 0, -1, 1, -1, -1, 1, 1, 1, 1, 1,
        -1, -1, 1, -1, 0, -1, 1, -1, -1, 1, 1, 1,
        1, 1, -1, -1, 1, -1, 0, -1, 1, -1,
        -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, 0, -1, 1, -1, -1, 1,
        1, 1, 1, 1, -1, -1, 1, -1, 0, -1, 1, -1,
        -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, 0, -1, 1, -1,
        1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1,
        -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, -1,
        1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1,
        -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1])
    data[11][3] = np.array([
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 0, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1,
        1, -1, 0, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1,
        1, 1, -1, 0, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1,
        1, -1, 1, -1, 0, -1, 1, -1, -1, 1, 1, 1, 1, -1,
        1, -1, -1, 1, -1, 0, -1, 1, -1, -1, 1, 1, 1, 1,
        1, 1, -1, -1, 1, -1, 0, -1, 1, -1, -1, 1, 1, 1,
        1, 1, 1, -1, -1, 1, -1, 0, -1, 1, -1, -1, 1, 1,
        1, 1, 1, 1, -1, -1, 1, -1, 0, -1, 1, -1, -1, 1,
        1, 1, 1, 1, 1, -1, -1, 1, -1, 0, -1, 1, -1, -1,
        1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 0, -1, 1, -1,
        1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1,
        1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, -1,
        1, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1])
    data[12][3] = np.array([
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 0, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1,
        1, -1, 0, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, -1,
        1, -1, -1, 0, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, -1,
        1, 1, -1, -1, 0, 1, 1, -1, 1, -1, 1, -1, -1, -1, 1,
        1, -1, 1, -1, -1, 0, 1, 1, 1, 1, -1, -1, -1, 1, 1,
        1, 1, -1, 1, -1, -1, 0, 1, 1, -1, -1, -1, 1, 1, -1,
        1, 1, 1, -1, 1, -1, -1, 0, 1, -1, -1, 1, 1, -1, 1,
        -1, 1, 1, 1, 1, 1, 1, 1, 0, -1, -1, -1, -1, -1, -1,
        -1, -1, 1, 1, -1, 1, -1, -1, 1, 0, -1, -1, 1, -1, 1,
        -1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 0, -1, -1, 1, -1,
        -1, 1, -1, 1, -1, -1, -1, 1, 1, 1, 1, 0, -1, -1, 1,
        -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, -1, -1,
        -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1,
        -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1,
        -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1])

    data[4][4] = np.array([
        0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, -1, 1, -1, -1,
        1, -1, 0, 1, 1, -1, 1, -1, 1, -1, -1, 0, 1, 1, -1, 1,
        1, 1, -1, -1, 1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, 1,
        1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, 1, -1, -1, 1,
        0, 0, 0, 0, -1, -1, -1, -1])
    data[5][4] = np.array([
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, -1, -1, -1, -1, 1, 1, 1,
        1, -1, 0, -1, 1, 1, -1, -1, 1, 1, -1, -1, 0, 1, 1, 1, 1, -1,
        1, -1, 1, 1, 0, -1, -1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1,
        1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, 1,
        1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1,
        0, 0, 0, 0, 0, -1, -1, -1, -1])
    data[6][4] = np.array([
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, -1, -1, -1, -1, 1, 1, 1, 1,
        1, -1, 0, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 0, 1, 1, 1, 1, -1, -1,
        1, -1, 1, 1, 0, -1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 0, 1, -1, 1, -1,
        1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1,
        1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, -1, 1,
        0, 0, 0, 0, 0, 0, -1, -1, -1, -1])
    data[7][4] = np.array([
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, -1, 1, 1, 1, -1, -1, -1,
        1, 1, -1, 0, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 0, 1, -1, 1, 1,
        1, -1, -1, 1, -1, 1, -1, 0, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1, -1, 0,
        1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 0, 1, -1, 1, 1, 1, 1, -1, -1,
        -1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 1, 1,
        1, -1, -1, -1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1,
        -1, 1, 1, 1, -1, -1, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1])
    data[8][4] = np.array([
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, -1, 1, 1, 1, -1, -1, -1,
        1, -1, 1, -1, 0, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 0, 1, -1,
        1, 1, 1, -1, -1, -1, 1, -1, 1, -1, 0, 1, -1, 1, 1, 1, -1, -1, 1, -1,
        -1, 1, -1, 0, 1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 0, 1, -1, 1,
        1, 1, 1, 1, -1, -1, -1, 1, -1, 0, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1,
        -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1,
        1, -1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1])
    data[9][4] = np.array([
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1,
        1, -1, 0, -1, 1, -1, -1, 1, 1, 1, 1, -1,
        -1, 1, 1, -1, 0, -1, 1, -1, -1, 1, 1, 1, 1, -1,
        1, -1, 1, -1, 0, -1, 1, -1, -1, 1, 1, 1, 1, 1,
        -1, -1, 1, -1, 0, -1, 1, -1, -1, 1, 1, 1,
        1, 1, -1, -1, 1, -1, 0, -1, 1, -1,
        -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, 0, -1, 1, -1, -1, 1,
        1, 1, 1, 1, -1, -1, 1, -1, 0, -1, 1, -1,
        -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1,
        1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1,
        -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, -1,
        1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1,
        -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1])
    data[10][4] = np.array([
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 0, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1,
        1, -1, 0, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1,
        1, 1, -1, 0, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1,
        1, -1, 1, -1, 0, -1, 1, -1, -1, 1, 1, 1, 1, -1,
        1, -1, -1, 1, -1, 0, -1, 1, -1, -1, 1, 1, 1, 1,
        1, 1, -1, -1, 1, -1, 0, -1, 1, -1, -1, 1, 1, 1,
        1, 1, 1, -1, -1, 1, -1, 0, -1, 1, -1, -1, 1, 1,
        1, 1, 1, 1, -1, -1, 1, -1, 0, -1, 1, -1, -1, 1,
        1, 1, 1, 1, 1, -1, -1, 1, -1, 0, -1, 1, -1, -1,
        1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1,
        1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1,
        1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, -1,
        1, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1])
    data[11][4] = np.array([
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 0, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1,
        1, -1, 0, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, -1,
        1, -1, -1, 0, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, -1,
        1, 1, -1, -1, 0, 1, 1, -1, 1, -1, 1, -1, -1, -1, 1,
        1, -1, 1, -1, -1, 0, 1, 1, 1, 1, -1, -1, -1, 1, 1,
        1, 1, -1, 1, -1, -1, 0, 1, 1, -1, -1, -1, 1, 1, -1,
        1, 1, 1, -1, 1, -1, -1, 0, 1, -1, -1, 1, 1, -1, 1,
        -1, 1, 1, 1, 1, 1, 1, 1, 0, -1, -1, -1, -1, -1, -1,
        -1, -1, 1, 1, -1, 1, -1, -1, 1, 0, -1, -1, 1, -1, 1,
        -1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 0, -1, -1, 1, -1,
        -1, 1, -1, 1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1,
        -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, -1, -1,
        -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1,
        -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1,
        -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1])
    data[12][4] = np.array([
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 0, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1,
        1, -1, 0, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, -1, -1,
        1, -1, -1, 0, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1,
        1, 1, -1, -1, 0, 1, 1, -1, 1, -1, 1, -1, -1, -1, 1, 1,
        1, -1, 1, -1, -1, 0, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1,
        1, 1, -1, 1, -1, -1, 0, 1, 1, -1, -1, -1, 1, 1, -1, 1,
        1, 1, 1, -1, 1, -1, -1, 0, 1, -1, -1, 1, 1, -1, 1, -1,
        -1, 1, 1, 1, 1, 1, 1, 1, 0, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, 1, 1, -1, 1, -1, -1, 1, 0, -1, -1, 1, -1, 1, 1,
        -1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 0, -1, -1, 1, -1, 1,
        -1, 1, -1, 1, -1, -1, -1, 1, 1, 1, 1, 0, -1, -1, 1, -1,
        -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, -1, -1, 1,
        -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1,
        -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1,
        -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1])


# ## std_ord function
def _std_ord(m):
    v1 = [x for x in range(0, m)]
    v2 = [x for x in range(m, 2*m)]
    ord1 = [2 * x - 1 for x in range(1, len(v1)+1)]
    ord2 = [2 * x for x in range(1, len(v2)+1)]
    return np.array(v1+v2)[np.argsort(ord1+ord2)]


def definitive_screening(m=0, c=0, center=0, randomize=False
                         ) -> pd.DataFrame:
    mess = "Definitive Screening Designs only exist for"

    if m < 4:
        raise TypeError(f"{mess} 4-12 3-level factors")
    if m > 12:
        raise TypeError(f"{mess} 4-12 3-level factors")
    if c < 0:
        raise TypeError(f"{mess} 0-4 2-level categorical factors")
    if c > 4:
        raise TypeError(f"{mess} 0-4 2-level categorical factors")
    if (c > 0) and (center > 0):
        mess1 = "Cannnot add center points to design"
        raise TypeError(f"{mess1} with 2-level categorical factors")

    a = dds.data[m][c]
    df = pd.DataFrame(a.reshape(int(len(a)/(m+c)), m+c),
                      columns=[chr(65+1) for i in range(m+c)])
    des = pd.concat([df, df*(-1)], axis=0).reset_index(drop=True)
    ord = _std_ord(len(df))
    des = des.iloc[ord]

    if c == 0:
        des = pd.concat(
            [des, pd.DataFrame([[0]*(m+c)], columns=des.columns)]
            ).reset_index(drop=True)
    if center > 0:
        cpr = [0]*len(des.columns)
        for _ in range(1, center+1):
            des = pd.concat(
                [des, pd.DataFrame([cpr], columns=des.columns)]
                ).reset_index(drop=True)
    if randomize is True:
        des = des.loc(
            np.random.choice([1 for i in range(1, len(des)+center+1)]))
    if center == 0:
        if randomize is True:
            des = des.loc(np.random.choice([i for i in range(1, len(des)+1)]))
    return des


if __name__ == "__main__":
    for y in range(0, 5):
        for x in range(4, 13):
            print(x, y)
            print(definitive_screening(x, y))
