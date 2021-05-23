import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plt_show(img, figsize=(24, 6), gray=True):
    """默认画灰度图"""
    plt.figure(figsize=figsize)
    plt.grid(False)
    if gray:
        plt.imshow(img, 'gray')
    else:
        plt.imshow(img)
    plt.show()
    pass


def plt_show_(img, figsize=(24, 6), gray=True):
    """默认画灰度图"""
    plt.figure(figsize=figsize)
    plt.grid(False)
    if gray:
        plt.imshow(img, 'gray')
    else:
        plt.imshow(img)
    plt.show()
    pass


def prt(*args):
    # print(*args)
    pass


class SegCircle:
    """找出图片中的培养皿并擦除其他部分"""

    def __call__(self, input_img):
        """
        args:
            input_img:array; rgb格式图像
        ret:
            seged_img:ndarray;  检测出的培养皿图片，格式RGB，
                尺寸固定到(nh, nw) = (900, 900 * (w / h))
        """
        # img = cv2.imread(img_path)
        img = self.resize(input_img)
        seged_img = self._seg_circle(img)
        return seged_img

    def _seg_circle(self, img):
        """
        args:
            img:ndarray; rgb格式图片
        """
        img_hw = img.shape[:2]

        circles = self.detect_circle(img)  # ; print('r = ', circles[0][2])
        center_hw, radius = (circles[0][1], circles[0][0]), (circles[0][2])  # 圆心格式：(y, x)  (h, w)  (行， 列)

        mask = self.dis_map(img_hw, center_hw, radius)
        mask = mask.astype(np.uint8)

        res = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)
        # plt_show(res, figsize=(12,12))
        return res

    @staticmethod
    def detect_circle(img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
        circles_ = cv2.HoughCircles(
            gaussian, cv2.HOUGH_GRADIENT, dp=1, minDist=500,
            param1=200, param2=75, minRadius=100, maxRadius=500
        )
        circles = circles_[0, :, :]
        circles = np.uint16(np.around(circles))
        return circles

    @staticmethod
    def dis_map(img_shape, center, r):
        img_h, img_w = img_shape
        ch, cw = center
        tensor_h, tensor_w = np.meshgrid(np.arange(img_h), np.arange(img_w), indexing='ij')
        tensor = np.zeros((img_h, img_w, 2))
        tensor_mask = np.zeros((img_h, img_w))
        tensor[:, :, 0] = tensor_h
        tensor[:, :, 1] = tensor_w
        distances = np.sqrt((tensor[:, :, 0] - ch) ** 2 + (tensor[:, :, 1] - cw) ** 2)
        tensor_mask[distances <= r] = 255
        return tensor_mask

    @staticmethod
    def resize(img, fixed_size=900):
        h, w, _ = img.shape  # opencv.readimg 格式(h, w)
        nh, nw = fixed_size, int(fixed_size * (w / h))
        img_resize = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)  # resize格式(w, h)
        return img_resize


def count_metric(count_num, labeled_num):
    count_num = np.array(count_num)
    labeled_num = np.array(labeled_num)
    error = np.abs(count_num - labeled_num) / labeled_num
    error = np.round(error, 2)
    return error


def count_record(config, batch, label, count_num, error):
    with open("count_log.txt", "a") as f:
        mean_error = np.array(error).mean().round(3)
        f.write("Image list:" + str(batch) + "\n")
        f.write("config:\n")
        f.write("min_cell_area:" + str(config["min_cell_area"]) + "|" +
                "max_cell_area:" + str(config["max_cell_area"]) + "\n")
        f.write("hsv_low:" + str(config["hsv_low"]) + "|" +
                "hsv_up:" + str(config["hsv_up"]) + "\n")
        f.write("\tmin_cell_area:" + str(config["min_cell_area"]) + "|" +
                "good_cell_area:" + str(config["good_cell_area"]) + "\n")
        f.write("\tadjust_ratio_low:" + str(config["adjust_ratio_low"]) + "|" +
                "adjust_ratio_up:" + str(config["adjust_ratio_up"]) + "\n")
        f.write("cell label num:" + str(label) + "\n")
        f.write("cell count num:" + str(count_num) + "\n")
        f.write("cell count err:" + str(error) + '\nmean error:' + str(mean_error) + "\n\n")


def hsv_dist_plot(hsv):
    plt.figure(figsize=(15, 8))
    sns.distplot(hsv[:, :, 0].flatten(), color="Y")
    sns.distplot(hsv[:, :, 1].flatten(), color="G")
    sns.distplot(hsv[:, :, 2].flatten(), color="Black")
    plt.show()


def hsv_select(img, hsv, h=[0, 255], s=[0, 255], v=[0, 255], show_mask=False):
    """拿到h[]， s[], v[]数值范围内的图像"""
    low_purple = np.array([h[0], s[0], v[0]])  # [h, s, v]
    high_purple = np.array([h[1], s[1], v[1]])
    mask = cv2.inRange(hsv, low_purple, high_purple)
    # print(mask.shape)  # 二维 二值（0/255）
    if show_mask:
        plt_show(mask, figsize=(15, 15))
    masked_image = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)
    plt_show(masked_image, figsize=(15, 15))
    return masked_image
