import cv2
import numpy as np
import pandas as pd
from collections import Counter
from cell_count_utils import SegCircle, hsv_dist_plot, hsv_select, plt_show, prt
from contour_analyser import PopuCntAnalyser
from hsv_agent import HsvImgCntAgent


def h_thresh_select(hsv_img, up_boundary=0.98):
    img_hsv_h = hsv_img[:, :, 0]
    # 拿到色调低于100的像素的面积
    _, h_res = cv2.threshold(img_hsv_h, 100, 255, cv2.THRESH_BINARY)
    img_pix = Counter(h_res.flatten())
    nan_roi_area = img_pix[0] / (img_pix[0] + img_pix[255])
    # 根据像素面积选择h通道的阈值
    img_hsv_h_pixel_array = img_hsv_h.flatten()
    pixel_counted = Counter(img_hsv_h_pixel_array)
    h_df = pd.DataFrame({"num": pixel_counted})
    h_df["ratio"] = h_df["num"].cumsum() / len(img_hsv_h_pixel_array)

    v_index = h_df[(h_df["ratio"] > nan_roi_area) & (h_df["ratio"] < up_boundary)].index.tolist()
    # print(v_index)
    return v_index[0], v_index[-1]


def s_thresh_value(rgb_img):
    img_gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    img_thresh = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    img_pix = Counter(img_thresh.flatten())
    area = img_pix[0] / (img_pix[0] + img_pix[255])
    return area


def s_thresh_select(hsv_img, low, up=1):
    v = hsv_img[:, :, 1].flatten()
    v_counted = Counter(v)
    v_df = pd.DataFrame({"num": v_counted})
    v_df["ratio"] = v_df["num"].cumsum() / len(v)

    v_index = v_df[(v_df["ratio"] > low) & (v_df["ratio"] < up)].index.tolist()
    # print(v_index)
    return v_index[0], v_index[-1]


def v_thresh_select(hsv_img, valid_area=(0.02, 0.98)):
    """选择面积占比为0.98-0.02的有效亮度区域"""
    v = hsv_img[:, :, 2].flatten()
    v_counted = Counter(v)
    v_df = pd.DataFrame({"num": v_counted})
    v_df["ratio"] = v_df["num"].cumsum() / len(v)

    low = v_df.iat[0, 1] + valid_area[0]
    up = valid_area[1]
    v_index = v_df[(v_df["ratio"] > low) & (v_df["ratio"] < up)].index.tolist()
    # print(v_index)
    return v_index[0], v_index[-1]


def roi_select(rgb_image):
    """

    :param rgb_image: 这里只接收分割培养皿后的图片
    :return:
    """
    # rgb_img = img_segor(rgb_img)

    img_hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # hsv_dist_plot(img_hsv)

    h_thresh = h_thresh_select(img_hsv)

    s_area_ = s_thresh_value(rgb_image)
    s_thresh = s_thresh_select(img_hsv, s_area_)

    v_thresh = v_thresh_select(img_hsv)

    prt('H:{} S:{} V:{}'.format(h_thresh, s_thresh, v_thresh))
    # selected_image_h1 = hsv_select(rgb_image, img_hsv, h=h_thresh, s=s_thresh, v=v_thresh)
    return h_thresh, s_thresh, v_thresh


def area_select(counters):
    cnt_area_list = []
    for cnt in counters:
        cnt_area_list.append(cv2.contourArea(cnt))
    cnt_area_list_array = np.array(cnt_area_list)
    min_area = np.percentile(cnt_area_list_array, 25)
    max_area = np.percentile(cnt_area_list_array, 75)
    return min_area, max_area


if __name__ == "__main__":

    img_file_list_1 = ['./data/cells1/187.jpg', './data/cells1/296.jpg', './data/cells1/308.jpg']
    img_file_list_2 = ["./data/cells2/177.jpg", "./data/cells2/216.jpg", "./data/cells2/219.jpg",
                       "./data/cells2/322.jpg", "./data/cells2/330.jpg", "./data/cells2/355.jpg"]
    img_list = img_file_list_1.extend(img_file_list_2)
    print(img_list)
    img_segor = SegCircle()
    # file_pth = img_file_list_1[0]
    for file_pth in img_file_list_2:
        print(file_pth)
        img_bgr = cv2.imread(file_pth)  # 读入图像并转换格式
        img_rgb_cvt = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        rgb_img = img_segor(img_rgb_cvt)
        h, s, v = roi_select(rgb_img)


    # img_rgb = img_segor(img_rgb_cvt)
    # plt_show(img_rgb, figsize=(15, 15))
    #
    # img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    # hsv_dist_plot(img_hsv)
    #
    # h_thresh = h_thresh_select(img_hsv)
    # s_area = s_thresh_value(img_rgb)
    # s_thresh = s_thresh_select(img_hsv, s_area)
    #
    # v_thresh = v_thresh_select(img_hsv)
    #
    # print('H:{} S:{} V:{}'.format(h_thresh, s_thresh, v_thresh))
    # selected_image_h1 = hsv_select(img_rgb, img_hsv, h=h_thresh, s=s_thresh, v=v_thresh)
