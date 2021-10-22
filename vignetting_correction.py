#!/usr/bin/env python

import argparse
from pathlib import Path
import numpy as np
import cv2


def center_of_mass(gray_img):
    gray_img = cv2.GaussianBlur(gray_img, ksize=(3, 3), sigmaX=10)
    pixel_sum = gray_img.sum()

    line_arr = np.arange(gray_img.shape[0])
    line_arr = np.repeat(line_arr[:, np.newaxis], gray_img.shape[1], axis=-1)

    col_arr = np.arange(gray_img.shape[1])
    col_arr = np.repeat(col_arr[np.newaxis, :], gray_img.shape[0], axis=0)

    return (
        int((line_arr * gray_img).sum() / pixel_sum),
        int((col_arr * gray_img).sum() / pixel_sum),
    )


def compute_r2(gray_img, cm):
    def distance(x, y):
        return (np.square(cm[0] - x) + np.square(cm[1] - y)) / (
            np.square(cm[0]) + np.square(cm[1])
        )

    return np.fromfunction(distance, gray_img.shape)


def check_conditions(v):
    a, b, c = v
    c1 = lambda a, b, c: (a > 0) and (b == c) and (c == 0)
    c2 = lambda a, b, c: (a >= 0) and (b > 0) and (c == 0)
    c3 = lambda a, b, c: (c == 0) and (b < 0) and (-a <= (2 * b))
    c4 = lambda a, b, c: (c > 0) and ((b * b) < (3 * a * c))
    c5 = lambda a, b, c: (c > 0) and ((b * b) == (3 * a * c)) and (b >= 0)
    c6 = lambda a, b, c: (c > 0) and ((b * b) == (3 * a * c)) and (-b >= (3 * c))

    qp = lambda a, b, c: (-2 * b + np.sqrt(4 * b * b - 12 * a * c)) / (6 * c)
    qm = lambda a, b, c: (-2 * b - np.sqrt(4 * b * b - 12 * a * c)) / (6 * c)
    c7 = lambda a, b, c: (c > 0) and ((b * b) > (3 * a * c)) and (qp(a, b, c) <= 0)
    c8 = lambda a, b, c: (c > 0) and ((b * b) > 3 * a * c) and (qm(a, b, c) >= 1)
    c9 = lambda a, b, c: (
        (c < 0)
        and ((b * b) > 3 * a * c)
        and (qp(a, b, c) >= 1)
        and (qm(a, b, c) <= 0)
        and (max(qm(a, b, c), qp(a, b, c)) >= 1)
    )
    return (
        c1(a, b, c)
        or c2(a, b, c)
        or c3(a, b, c)
        or c4(a, b, c)
        or c5(a, b, c)
        or c6(a, b, c)
        or c7(a, b, c)
        or c8(a, b, c)
        or c9(a, b, c)
    )


def gain(v, r2):
    r4 = r2 * r2
    r6 = r4 * r2
    return 1 + v[0] * r2 + v[1] * r4 + v[2] * r6


def compute_entropy(img, radius=4):
    log_img = 255 * np.log(1 + img) / 8
    hist = np.zeros((256,), dtype="float")
    flattened = np.ravel(log_img)
    k_d = np.floor(flattened).astype("int")
    k_u = np.ceil(flattened).astype("int")
    np.add.at(hist, k_d, 1 + k_d - flattened)
    np.add.at(hist, k_u, k_u - flattened)

    tmp_hist = np.zeros((256 + radius * 2,))
    tmp_hist[:radius] = hist[1 : radius + 1][::-1]
    tmp_hist[256 + radius :] = hist[256 - radius - 1 : -1 :][::-1]
    tmp_hist[radius : 256 + radius] = hist
    strides = np.lib.stride_tricks.as_strided(
        tmp_hist, (256, 2 * radius + 1), 2 * tmp_hist.strides
    ) * np.array([1, 2, 3, 4, 5, 4, 3, 2, 1 / 25.0])
    hist = np.sum(
        strides,
        axis=1,
    )
    pk = hist / hist.sum()
    nz_pk = pk[~np.isclose(pk, 0)]
    return -np.sum(nz_pk * np.log(nz_pk))


def vignetting_correction(gray_img, speedup=False):
    abc = np.array([0,0,0])
    delta = 8
    og_img = gray_img
    r2_og = compute_r2(og_img, center_of_mass(og_img))
    r2 = r2_og

    if (speedup):
        gray_img = cv2.resize(gray_img, (100, int(100 / gray_img.shape[1] * gray_img.shape[0])))
        r2 = compute_r2(gray_img, center_of_mass(gray_img))

    hm = compute_entropy(gray_img)
    f = gray_img.copy()

    

    while delta > (1 / 256):
        varr = [
            abc + np.array([delta, 0, 0]),
            abc + np.array([-delta, 0, 0]),
            abc + np.array([0, delta, 0]),
            abc + np.array([0, -delta, 0]),
            abc + np.array([0, 0, delta]),
            abc + np.array([0, 0, -delta]),
        ]
        entropies = [hm] * len(varr)
        for i, v in enumerate(varr):
            if check_conditions(v):
                new_img = gray_img * gain(v, r2)
                entropies[i] = compute_entropy(new_img)

        h_idx = np.argmin(entropies)
        h = entropies[h_idx]
        if h < hm:
            hm = h
            abc = varr[h_idx]
            f = og_img * gain(abc, r2_og)
            delta = 8
        else:
            delta /= 2
    print("Final a, b, c :", abc[0], abc[1], abc[2])
    return f


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Correct vignetting for a given picture"
    )
    parser.add_argument("PICTURE_PATH", type=str)
    parser.add_argument("--speed", action="store_true", help="Speedup the method by using a downscaled image to compute a, b, c")
    args = parser.parse_args()
    img_path = Path(args.PICTURE_PATH)

    im = cv2.imread(args.PICTURE_PATH)

    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
    gray = yuv[:, :, 0].astype("float")    

    res = vignetting_correction(gray, args.speed)
    res = (res - res.min()) / ((res - res.min()).max()) * 255

    yuv[:, :, 0] = res.astype("uint8")
    out = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    cv2.imwrite(img_path.stem + "_corrected" + img_path.suffix, out)
