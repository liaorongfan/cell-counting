"""
# -*- coding: utf-8 -*-
# CreateBy: liaorongfan
# CreateAT: 2020/9/15
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from cell_count_utils import plt_show, prt


class HsvImgCntAgentAlt:

    default_config = {
        "hsv_low": [100, 20, 50],
        "hsv_up": [200, 250, 150],
        "initial_filtered_cnts_area": 15,  # 这个值在10 或 20 没有关系  这个参数用于在第一步中过滤杂质 不用于颜色轮廓过滤
        "count_mean_area": 1400
    }  # 这是一个全局的信息，hsv_low - hsv_up

    def __init__(self, img, config=None):
        """
        初始化时拿到用默认阈值分割的轮廓
        :param img: rgb格式图像
        """
        self.img = img
        if config is not None:
            self.config = config
        self.config = self.default_config
        self.cnts = self._initial_process(self.config)  # 初始化时都自动调用默认参数

        self.draw_index()  # 在图上显示轮廓的索引 test used

    def _initial_process(self, config):
        """处理流程
        1)颜色分割 --> 2）轮廓过滤
        """
        mask = self._get_mask_hsv_range(self.img, low=config["hsv_low"], up=config["hsv_up"])
        plt_show(mask, figsize=(15, 15))  # test used

        contours, _ = cv2.findContours(  # 拿到轮廓 contours:list[ndarray]
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        prt("初始HSV阈值检测颜色轮廓个数：", len(contours))
        prt("initial_filtered_cnts_area", config["initial_filtered_cnts_area"])
        selected_cnts = self.filter_cnts(contours, min_area=config["initial_filtered_cnts_area"])  # 初步过滤掉一些轮廓
        prt("初步过滤后的颜色轮廓个数：", len(selected_cnts))

        return selected_cnts

    @staticmethod
    def _get_mask_hsv_range(img, low, up):
        """这里返回的轮廓是后续群落计数的基础
        直接返回过滤后的全图轮廓
        """
        # 1 颜色转换
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # 2 阈值过滤 选出轮廓
        hsv_low = np.array(low)
        hsv_high = np.array(up)
        mask = cv2.inRange(img_hsv, hsv_low, hsv_high)
        # 3 膨胀处理二值轮廓掩膜  这一步的处理 抑制了培养皿边缘的噪音
        kernel = np.ones((3, 3), np.uint8)
        mask_dilate = cv2.dilate(mask, kernel, iterations=1)
        # 4 返回过滤后的二值轮廓图像
        return mask_dilate

    def adjust_hsv_threshold(self, image, config):  # 注意字典类型的变量，传递的是引用，即后续操作改变了这个变量的值，前面的值也会变, 可以把它看成一个全局变量

        img = image.copy()  # 此函数内的img变量是参数image的一个副本，img的改变不会影响原变量image的值
        satisfied = False
        mask = self._get_mask_hsv_range(img, low=config["hsv_low"], up=config["hsv_up"])
        # plt_show(mask, figsize=(15, 15))  # test used

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 拿到轮廓...

        try:
            prt("\n|---------------------------------------> 自适应阈值调节 <-------------------------------------|")
            prt("HSV配置参数: hsv_low={}, hsv_up={}".format(config['hsv_low'], config["hsv_up"]))
            sorted_cnts_area = sorted(map(cv2.contourArea, contours))  # 计算并排序上述轮廓的面积
            prt('轮廓面积：', sorted_cnts_area)

            min_cell_area, max_cell_area = config["min_cell_area"], config["max_cell_area"]
            adjust_ratio_low, adjust_ratio_up = config["adjust_ratio_low"], config["adjust_ratio_up"]

            if sorted_cnts_area[0] < min_cell_area:  # 最小的轮廓面积小于标定面积config["min_cell_area"]，则反馈调整阈值
                min_area_err = sorted_cnts_area[0] - min_cell_area
                config["hsv_low"] = self.adjust_hsv_config(config["hsv_low"], adjust_ratio_low, min_area_err)  # 更新下限的值
                prt('小面积误差修正: 误差:[{}];阈值下限调整为[{}]'.format(min_area_err, config["hsv_low"]))  # 正负的意义...

            if sorted_cnts_area[-1] > max_cell_area:  # 最大的轮廓面积大于标定面积config["max_cell_area"]，则反馈调整阈值
                max_area_err = sorted_cnts_area[-1] - max_cell_area
                config["hsv_up"] = self.adjust_hsv_config(config["hsv_up"], adjust_ratio_up, max_area_err)  # 更新上限的值
                prt('大面积误差修正: 误差:[{}];阈值上限调整为[{}]'.format(max_cell_area, config["hsv_up"]))

            if (sorted_cnts_area[0] > min_cell_area) & (sorted_cnts_area[-1] < max_cell_area):
                prt('各轮廓均满足面积条件,自适应反馈调整结束')
                satisfied = True

        except (ValueError, IndexError):
            prt('HSV阈值范围[{}，{}]下没有轮廓被选出'.format(config["hsv_low"], config["hsv_up"]))

        return satisfied, contours, config

    @staticmethod
    def adjust_hsv_config(hsv_config, adjust_ratio, err):
        """
        根据误差err和调整率adjust_ratio驱动颜色阈值hsv_config
        :param hsv_config:list in dict; 颜色阈值
        :param adjust_ratio: float; 调整率
        :param err: float； 面积误差
        :return: hsv_config; 调整后的颜色阈值
        """
        hsv = np.array(hsv_config)
        ratio = np.array(adjust_ratio)  # 0.5是为了拉低误差
        hsv = hsv - ratio * err * 0.5  # 这里默认一个相关关系，最小的轮廓面积若小于标定面积，即min_area_err < 0, 就提高hsv_low下限，增加检测概率...
        hsv = hsv.astype(np.int32)      # ...若最大的轮廓面积大于标定面积，拉低hsv_up限线的阈值，增加轮廓检出的概率

        hsv[hsv < 10] = 10  # 下确界20
        hsv[hsv > 255] = 200  # 上确界200
        return list(hsv)

    @staticmethod
    def filter_cnts(contours, min_area=20):
        """初步过滤掉一些噪音轮廓
        这里可以进一步添加规则 得到较优质的轮廓
        """
        filtered_cnts = []
        for cnt in contours:
            cnt_area = cv2.contourArea(cnt)
            if cnt_area > min_area:
                filtered_cnts.append(cnt)

        return filtered_cnts

    @staticmethod
    def img_field_select(img, low, up):
        # 1 颜色转换
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # 2 阈值过滤 选出轮廓
        hsv_low = np.array(low)
        hsv_high = np.array(up)
        mask = cv2.inRange(img_hsv, hsv_low, hsv_high)
        # 3 抹掉没有选中的区域
        res = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)
        return res

    def draw_index(self):
        """画出轮廓的索引便于逐个分析轮廓"""
        img_draw_temp = self.img.copy()  # 将轮廓画在图像的副本上，并显示图像的副本，不影响原图self.img。
        cv2.drawContours(img_draw_temp, self.cnts, -1, (255, 255, 0), 1)
        for idx, cnt in enumerate(self.cnts):  # 这里索引的顺序是self.cnts的顺序
            try:
                cv2.drawContours(img_draw_temp, cnt, -1, (255, 255, 0), 1)
                M = cv2.moments(cnt)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(img_draw_temp, (cx, cy), 2, (0, 255, 0), 1)
                cv2.putText(img_draw_temp,
                            str(idx),
                            (cx + 5, cy + 5),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            0.8, (0, 255, 255), 1)
            except Exception:
                print("missing %d" % idx)
        plt_show(img_draw_temp, figsize=(15, 15))

    def adjust_hsv_(self, img, config):
        img = img.copy()
        satisfied = False
        min_cell_area, max_cell_area = config["min_cell_area"], config["max_cell_area"]
        prt(min_cell_area, max_cell_area)

        mask = self._get_mask_hsv_range(img, low=config["hsv_low"], up=config["hsv_up"])
        plt_show(mask, figsize=(15, 15))  # test used
        # 拿到轮廓
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
        plt_show(img, figsize=(15, 15))  # test used

        try:
            sorted_cnts_area = sorted(map(cv2.contourArea, contours)); print('轮廓面积：', sorted_cnts_area)
            if (sorted_cnts_area[0] < min_cell_area) & (sorted_cnts_area[-1] > max_cell_area):
                adjust_ratio_low, adjust_ratio_up = config["adjust_ratio_low"], config["adjust_ratio_up"]
                hsv_low, hsv_up = np.array(config["hsv_low"]), np.array(config["hsv_up"])

                min_area_err = sorted_cnts_area[0] - min_cell_area; print(min_area_err)  # 正负是有意义的
                max_area_err = sorted_cnts_area[-1] - max_cell_area; print(max_area_err)

                hsv_low = hsv_low - adjust_ratio_low * min_area_err
                hsv_up = hsv_up + adjust_ratio_up * max_area_err

                hsv_low[hsv_low < 10] = 10  # 下确界20
                hsv_up[hsv_up > 255] = 200  # 上确界 255 防止越界

                hsv_low = hsv_low.astype(np.int32)
                hsv_up = hsv_up.astype(np.int32)

                config["hsv_low"] = list(hsv_low)
                config["hsv_up"] = list(hsv_up)
            else:
                satisfied = True
        except (ValueError, IndexError):
            print(contours)
        return satisfied, contours

    def adjust_hsv_threshold_(self, image, config):  # 注意字典类型的变量，传递的是引用，即后续操作改变了这个变量的值，前面的值也会变, 可以把它看成一个全局变量
        img = image.copy()  # 此函数内的img变量是参数image的一个副本，img的改变不会影响原变量image的值
        satisfied = False
        mask = self._get_mask_hsv_range(img, low=config["hsv_low"], up=config["hsv_up"])
        # plt_show(mask, figsize=(15, 15))  # test used

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 拿到轮廓...

        try:
            prt("----------------自适应阈值调节---------------")
            sorted_cnts_area = sorted(map(cv2.contourArea, contours))  # 计算并排序上述轮廓的面积
            prt('轮廓面积：', sorted_cnts_area)

            min_cell_area, max_cell_area = config["min_cell_area"], config["max_cell_area"]
            adjust_ratio_low, adjust_ratio_up = config["adjust_ratio_low"], config["adjust_ratio_up"]
            hsv_low, hsv_up = np.array(config["hsv_low"]), np.array(config["hsv_up"])

            if sorted_cnts_area[0] < min_cell_area:  # 最小的轮廓面积小于标定面积config["min_cell_area"]，则反馈调整阈值
                min_area_err = sorted_cnts_area[0] - min_cell_area
                hsv_low = hsv_low - adjust_ratio_low * min_area_err  # ...这里默认一个相关关系，最小的轮廓面积若小于标定面积，即min_area_err < 0, 就提高hsv_low下限，增加检测概率
                hsv_low[hsv_low < 20] = 20  # 下确界20
                hsv_low = hsv_low.astype(np.int32)
                config["hsv_low"] = list(hsv_low)  # 更新下限的值
                prt(config["hsv_low"])

            if sorted_cnts_area[-1] > max_cell_area:  # 最大的轮廓面积大于标定面积config["max_cell_area"]，则反馈调整阈值
                max_area_err = sorted_cnts_area[-1] - max_cell_area
                hsv_up = hsv_up - adjust_ratio_up * max_area_err
                hsv_up[hsv_up > 255] = 200  # 上确界 255 防止越界
                hsv_up = hsv_up.astype(np.int32)
                config["hsv_up"] = list(hsv_up)  # 更新上限的值
                prt(config["hsv_up"])

            if (sorted_cnts_area[0] > min_cell_area) & (sorted_cnts_area[-1] < max_cell_area):
                prt('各轮廓均满足面积条件,自适应反馈调整结束')
                satisfied = True

        except (ValueError, IndexError):
            prt('空列表：', contours, "config:", config["hsv_low"], config["hsv_up"])
            pass

        return satisfied, contours, config


class HsvImgCntAgent:
    default_config = {
        "hsv_low": [100, 20, 50],
        "hsv_up": [200, 250, 200],  # cells2 ：v 用200， cells1: v用150
        "initial_filtered_cnts_area": 10,  # 这个参数用于在第一步中过滤杂质 不用于颜色轮廓过滤
        "count_mean_area": 1400
    }  # 这是一个全局的信息，hsv_low - hsv_up

    def __init__(self, img, config=None):
        """
        初始化时拿到用默认阈值分割的轮廓
        :param img: rgb格式图像
        """
        self.img = img
        if config is not None:
            self.config = config
        else:
            self.config = self.default_config
        self.cnts = self._initial_process(self.config)  # 初始化时都自动调用默认参数

        self.draw_index()  # 在图上显示轮廓的索引 test used

    def _initial_process(self, config):
        """处理流程
        1)颜色分割 --> 2）轮廓过滤
        """
        mask = self._get_mask_hsv_range(self.img, low=config["hsv_low"], up=config["hsv_up"])
        prt("初始HSV阈值分割出来的轮廓：")
        plt_show(mask, figsize=(15, 15))  # test used

        contours, _ = cv2.findContours(  # 拿到轮廓 contours:list[ndarray]
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        prt("初始HSV阈值检测颜色轮廓个数：", len(contours))
        prt("initial_filtered_cnts_area", config["initial_filtered_cnts_area"])
        prt("HSV_low", config["hsv_low"])
        selected_cnts = self.filter_cnts(contours, min_area=config["initial_filtered_cnts_area"])  # 初步过滤掉一些轮廓
        prt("初步过滤后的颜色轮廓个数：", len(selected_cnts))

        return selected_cnts

    @staticmethod
    def _get_mask_hsv_range(img, low, up):
        """这里返回的轮廓是后续群落计数的基础
        直接返回过滤后的全图轮廓
        """
        # 1 颜色转换
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # 2 阈值过滤 选出轮廓
        hsv_low = np.array(low)
        hsv_high = np.array(up)
        mask = cv2.inRange(img_hsv, hsv_low, hsv_high)
        # 3 膨胀处理二值轮廓掩膜  这一步的处理 抑制了培养皿边缘的噪音
        kernel = np.ones((3, 3), np.uint8)
        mask_dilate = cv2.dilate(mask, kernel, iterations=1)
        # 4 返回过滤后的二值轮廓图像
        return mask_dilate

    def adjust_hsv_threshold(self, image, config):  # 注意字典类型的变量，传递的是引用，即后续操作改变了这个变量的值，前面的值也会变, 可以把它看成一个全局变量

        img = image.copy()  # 此函数内的img变量是参数image的一个副本，img的改变不会影响原变量image的值
        satisfied = False
        mask = self._get_mask_hsv_range(img, low=config["hsv_low"], up=config["hsv_up"])
        # plt_show(mask, figsize=(15, 15))  # test used

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 拿到轮廓...

        try:
            prt("\n|---------------------------------------> 自适应阈值调节 <-------------------------------------|")
            prt("HSV配置参数: hsv_low={}, hsv_up={}".format(config['hsv_low'], config["hsv_up"]))
            sorted_cnts_area = sorted(map(cv2.contourArea, contours))  # 计算并排序上述轮廓的面积
            prt('轮廓面积：', sorted_cnts_area)

            min_cell_area, max_cell_area = config["min_cell_area"], config["max_cell_area"]
            adjust_ratio_low, adjust_ratio_up = config["adjust_ratio_low"], config["adjust_ratio_up"]

            if sorted_cnts_area[0] < min_cell_area:  # 最小的轮廓面积小于标定面积config["min_cell_area"]，则反馈调整阈值
                min_area_err = sorted_cnts_area[0] - min_cell_area
                error = np.sign(min_area_err) * np.log(np.abs(min_area_err))
                config["hsv_low"] = self.adjust_hsv_config(config["hsv_low"], adjust_ratio_low, error)  # 更新下限的值
                prt('小面积误差修正: 误差:[{}];阈值下限调整为[{}]'.format(min_area_err, config["hsv_low"]))  # 正负的意义...

            if sorted_cnts_area[-1] > max_cell_area:  # 最大的轮廓面积大于标定面积config["max_cell_area"]，则反馈调整阈值
                max_area_err = sorted_cnts_area[-1] - max_cell_area
                error = np.sign(max_area_err) * np.log(np.abs(max_area_err))
                config["hsv_up"] = self.adjust_hsv_config(config["hsv_up"], adjust_ratio_up, error)  # 更新上限的值
                prt('大面积误差修正: 误差:[{}];阈值上限调整为[{}]'.format(max_cell_area, config["hsv_up"]))

            if (sorted_cnts_area[0] > min_cell_area) & (sorted_cnts_area[-1] < max_cell_area):
                prt('各轮廓均满足面积条件,自适应反馈调整结束')
                satisfied = True

        except (ValueError, IndexError):
            prt('HSV阈值范围[{}，{}]下没有轮廓被选出'.format(config["hsv_low"], config["hsv_up"]))

        return satisfied, contours, config

    @staticmethod
    def adjust_hsv_config(hsv_config, adjust_ratio, err):
        """
        根据误差err和调整率adjust_ratio驱动颜色阈值hsv_config
        :param hsv_config:list in dict; 颜色阈值
        :param adjust_ratio: float; 调整率
        :param err: float； 面积误差
        :return: hsv_config; 调整后的颜色阈值
        """
        hsv = np.array(hsv_config)
        ratio = np.array(adjust_ratio)
        hsv = hsv - ratio * err   # 这里默认一个相关关系，最小的轮廓面积若小于标定面积，即min_area_err < 0, 就提高hsv_low下限，增加检测概率...
        hsv = hsv.astype(np.int32)  # ...若最大的轮廓面积大于标定面积，拉低hsv_up限线的阈值，增加轮廓检出的概率

        hsv[hsv < 20] = 20  # 下确界20
        hsv[hsv > 255] = 200  # 上确界200
        return list(hsv)

    @staticmethod
    def filter_cnts(contours, min_area=20):
        """初步过滤掉一些噪音轮廓
        这里可以进一步添加规则 得到较优质的轮廓
        """
        filtered_cnts = []
        for cnt in contours:
            cnt_area = cv2.contourArea(cnt)
            if cnt_area > min_area:
                filtered_cnts.append(cnt)

        return filtered_cnts

    @staticmethod
    def img_field_select(img, low, up):
        # 1 颜色转换
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # 2 阈值过滤 选出轮廓
        hsv_low = np.array(low)
        hsv_high = np.array(up)
        mask = cv2.inRange(img_hsv, hsv_low, hsv_high)
        # 3 抹掉没有选中的区域
        res = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)
        return res

    def draw_index(self):
        """画出轮廓的索引便于逐个分析轮廓"""
        img_draw_temp = self.img.copy()  # 将轮廓画在图像的副本上，并显示图像的副本，不影响原图self.img。
        cv2.drawContours(img_draw_temp, self.cnts, -1, (255, 255, 0), 1)
        for idx, cnt in enumerate(self.cnts):  # 这里索引的顺序是self.cnts的顺序
            try:
                cv2.drawContours(img_draw_temp, cnt, -1, (255, 255, 0), 1)
                M = cv2.moments(cnt)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(img_draw_temp, (cx, cy), 2, (0, 255, 0), 1)
                cv2.putText(img_draw_temp,
                            str(idx),
                            (cx + 5, cy + 5),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            0.8, (0, 255, 255), 1)
            except Exception:
                print("missing %d" % idx)
        plt_show(img_draw_temp, figsize=(15, 15))
        plt.imsave('s-{}_min_area-{}-filter.jpg'
                   .format(self.default_config["hsv_low"][1],
                           self.default_config["initial_filtered_cnts_area"]),
                   img_draw_temp)

    def draw_single_cnts(self, index):
        img_draw_temp = self.img.copy()  # 将轮廓画在图像的副本上，并显示图像的副本，不影响原图self.img。
        cnt = self.cnts[index]
        try:
            cv2.drawContours(img_draw_temp, cnt, -1, (255, 255, 0), 1)

            rect = cv2.minAreaRect(cnt)  # 画出最小外接圆看一看
            box = cv2.boxPoints(rect)
            box = np.int0(box)  # int0 等价与int64 将浮点数转换为整数
            cv2.drawContours(img_draw_temp, [box], 0, (255, 0, 0), 2)

            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(img_draw_temp, (cx, cy), 2, (0, 255, 0), 1)
            cv2.putText(img_draw_temp,
                        str(index),
                        (cx + 5, cy + 5),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        0.8, (0, 255, 255), 1)
        except Exception:
            print("missing %d" % index)
        plt_show(img_draw_temp, figsize=(15, 15))

    def adjust_hsv_(self, img, config):
        img = img.copy()
        satisfied = False
        min_cell_area, max_cell_area = config["min_cell_area"], config["max_cell_area"]
        prt(min_cell_area, max_cell_area)

        mask = self._get_mask_hsv_range(img, low=config["hsv_low"], up=config["hsv_up"])
        plt_show(mask, figsize=(15, 15))  # test used
        # 拿到轮廓
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
        plt_show(img, figsize=(15, 15))  # test used

        try:
            sorted_cnts_area = sorted(map(cv2.contourArea, contours));
            print('轮廓面积：', sorted_cnts_area)
            if (sorted_cnts_area[0] < min_cell_area) & (sorted_cnts_area[-1] > max_cell_area):
                adjust_ratio_low, adjust_ratio_up = config["adjust_ratio_low"], config["adjust_ratio_up"]
                hsv_low, hsv_up = np.array(config["hsv_low"]), np.array(config["hsv_up"])

                min_area_err = sorted_cnts_area[0] - min_cell_area;
                print(min_area_err)  # 正负是有意义的
                max_area_err = sorted_cnts_area[-1] - max_cell_area;
                print(max_area_err)

                hsv_low = hsv_low - adjust_ratio_low * min_area_err
                hsv_up = hsv_up + adjust_ratio_up * max_area_err

                hsv_low[hsv_low < 20] = 20  # 下确界20
                hsv_up[hsv_up > 255] = 200  # 上确界 255 防止越界

                hsv_low = hsv_low.astype(np.int32)
                hsv_up = hsv_up.astype(np.int32)

                config["hsv_low"] = list(hsv_low)
                config["hsv_up"] = list(hsv_up)
            else:
                satisfied = True
        except (ValueError, IndexError):
            print(contours)
        return satisfied, contours

    def adjust_hsv_threshold_(self, image, config):  # 注意字典类型的变量，传递的是引用，即后续操作改变了这个变量的值，前面的值也会变, 可以把它看成一个全局变量
        img = image.copy()  # 此函数内的img变量是参数image的一个副本，img的改变不会影响原变量image的值
        satisfied = False
        mask = self._get_mask_hsv_range(img, low=config["hsv_low"], up=config["hsv_up"])
        # plt_show(mask, figsize=(15, 15))  # test used

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 拿到轮廓...

        try:
            prt("----------------自适应阈值调节---------------")
            sorted_cnts_area = sorted(map(cv2.contourArea, contours))  # 计算并排序上述轮廓的面积
            prt('轮廓面积：', sorted_cnts_area)

            min_cell_area, max_cell_area = config["min_cell_area"], config["max_cell_area"]
            adjust_ratio_low, adjust_ratio_up = config["adjust_ratio_low"], config["adjust_ratio_up"]
            hsv_low, hsv_up = np.array(config["hsv_low"]), np.array(config["hsv_up"])

            if sorted_cnts_area[0] < min_cell_area:  # 最小的轮廓面积小于标定面积config["min_cell_area"]，则反馈调整阈值
                min_area_err = sorted_cnts_area[0] - min_cell_area
                hsv_low = hsv_low - adjust_ratio_low * min_area_err  # ...这里默认一个相关关系，最小的轮廓面积若小于标定面积，即min_area_err < 0, 就提高hsv_low下限，增加检测概率
                hsv_low[hsv_low < 20] = 20  # 下确界20
                hsv_low = hsv_low.astype(np.int32)
                config["hsv_low"] = list(hsv_low)  # 更新下限的值
                prt(config["hsv_low"])

            if sorted_cnts_area[-1] > max_cell_area:  # 最大的轮廓面积大于标定面积config["max_cell_area"]，则反馈调整阈值
                max_area_err = sorted_cnts_area[-1] - max_cell_area
                hsv_up = hsv_up - adjust_ratio_up * max_area_err
                hsv_up[hsv_up > 255] = 200  # 上确界 255 防止越界
                hsv_up = hsv_up.astype(np.int32)
                config["hsv_up"] = list(hsv_up)  # 更新上限的值
                prt(config["hsv_up"])

            if (sorted_cnts_area[0] > min_cell_area) & (sorted_cnts_area[-1] < max_cell_area):
                prt('各轮廓均满足面积条件,自适应反馈调整结束')
                satisfied = True

        except (ValueError, IndexError):
            prt('空列表：', contours, "config:", config["hsv_low"], config["hsv_up"])
            pass

        return satisfied, contours, config
