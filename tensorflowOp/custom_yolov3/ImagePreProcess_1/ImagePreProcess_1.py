"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0
"""
import sys
from PIL import Image
import numpy as np


def convert(para1, para2, para3, para4):
    return (((para4 * 256 + para3) * 256 + para2) * 256) + para1


def read_yuv_file(filename):
    file_p = open(filename, 'rb')
    # the last 32 bytes are info about yuv file which is used to transfer jpg
    file_p.seek(-32,
            2)
    resize_width = convert(ord(file_p.read(1)), ord(file_p.read(1)), ord(file_p.read(1)),
                           ord(file_p.read(1)))
    resize_height = convert(ord(file_p.read(1)), ord(file_p.read(1)), ord(file_p.read(1)),
                            ord(file_p.read(1)))
    preprocess_width = convert(ord(file_p.read(1)), ord(file_p.read(1)),
                               ord(file_p.read(1)), ord(file_p.read(1)))
    preprocess_height = convert(ord(file_p.read(1)), ord(file_p.read(1)),
                                ord(file_p.read(1)), ord(file_p.read(1)))
    frame_id = convert(ord(file_p.read(1)), ord(file_p.read(1)), ord(file_p.read(1)),
                       ord(file_p.read(1)))

    file_p.seek(0, 0)
    uv_width = preprocess_width // 2
    uv_height = preprocess_height // 2

    Y_p = np.zeros((preprocess_height, preprocess_width), dtype=np.uint8)
    U_p = np.zeros((uv_height, uv_width), dtype=np.uint8)
    V_p = np.zeros((uv_height, uv_width), dtype=np.uint8)

    for m in range(preprocess_height):
        for n in range(preprocess_width):
            Y_p[m, n] = ord(file_p.read(1))
    for m in range(uv_height):
        for n in range(uv_width):
            U_p[m, n] = ord(file_p.read(1))
            V_p[m, n] = ord(file_p.read(1))

    file_p.close()
    return (Y_p, U_p,
            V_p), resize_width, resize_height, preprocess_width, \
           preprocess_height, frame_id


def yuv2rgb(Y_p, U_p, V_p, width, height):
    U_p = np.repeat(U_p, 2, 0)
    U_p = np.repeat(U_p, 2, 1)
    V_p = np.repeat(V_p, 2, 0)
    V_p = np.repeat(V_p, 2, 1)
    r_f = np.zeros((height, width), float, 'C')
    g_f = np.zeros((height, width), float, 'C')
    b_f = np.zeros((height, width), float, 'C')

    r_f = Y_p + 1.403 * (V_p - 128.0)
    g_f = Y_p - 0.343 * (U_p - 128.0) - 0.714 * (V_p - 128.0)
    b_f = Y_p + 1.77 * (U_p - 128.0)

    for m in range(height):
        for n in range(width):
            if (r_f[m, n] > 255):
                r_f[m, n] = 255
            if (g_f[m, n] > 255):
                g_f[m, n] = 255
            if (b_f[m, n] > 255):
                b_f[m, n] = 255
            if (r_f[m, n] < 0):
                r_f[m, n] = 0
            if (g_f[m, n] < 0):
                g_f[m, n] = 0
            if (b_f[m, n] < 0):
                b_f[m, n] = 0
    res_r = r_f.astype(np.uint8)
    res_g = g_f.astype(np.uint8)
    res_b = b_f.astype(np.uint8)
    return (res_r, res_g, res_b)


def tranfer_yue2_jpg(yuv_file):
    data, resize_width, resize_height, preprocess_width, preprocess_height, \
    frame_id = read_yuv_file(yuv_file)
    image_name = yuv_file.replace('_preprocessYUV', '_preProcess.jpg')

    im_rgb = np.zeros((3, preprocess_height, preprocess_width))
    im_rgb[0], im_rgb[1], im_rgb[2] = yuv2rgb(data[0], data[1], data[2],
                                              preprocess_width,
                                              preprocess_height)
    im_rgb = np.transpose(im_rgb, (1, 2, 0))

    image = Image.fromarray(im_rgb.astype(np.uint8), mode='RGB')
    image = image.crop((0, 0, resize_width, resize_height))
    image.save(image_name)


if __name__ == '__main__':
    param = sys.argv[1]  # raw image path
    yuv_list = param.split(',')
    for yuv_file in yuv_list:
        if "_preprocessYUV" in yuv_file:
            tranfer_yue2_jpg(yuv_file)
