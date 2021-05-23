"""
# -*- coding: utf-8 -*-
# CreateBy: liaorongfan
# CreateAT: 2020/9/15
"""
import cv2
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from cell_count_utils import plt_show, plt_show_, prt


class PopuCntAnalyserAlt:
    """
    分析每个轮廓中一共有几个群落
    以轮廓贯穿处理流程：
        1）自适应颜色阈值 逐个处理用默认阈值筛选出的轮廓
        2）对处理后的颜色轮廓做形状分析
            先画一些辅助图以bin_开头
            没有连接在一起的群落直接计数
            连在一起的群落用“计数轮廓”算细胞个数
    """
    def __init__(self, img_src, config, agent=None, adaptive=True):
        """

        :param counters: cv.findContours); 出入进来初步过滤后的轮廓
        :param img_src: ndarray; 处理过的图像，这里是分割培养皿后的图片
        :param config:
        :param agent:
        :param adaptive:
        """
        self.cnts = agent.cnts  # 传进来所有的轮廓
        self.img = img_src.copy()
        self.config = config
        self.config_temp = config
        self.agent = agent
        self.adaptive = adaptive
        self.total_popu = 0
        self.h, self.w, _ = img_src.shape

    def __getitem__(self, index):
        img_draw = self.img  # 每次都把轮廓画到同一张图上
        cnt = self.cnts[index]  # 轮廓

        cnts_num, adaptive_cnts = self.adaptive_hsv_counter(cnt)

        if cnts_num != 0:
            for cnt in adaptive_cnts:
                popu_num = self.cnts_counting(cnt)
                self.total_popu += popu_num
                self.draw_number(img_draw, cnt, popu_num)

    def getitem_test(self, index):
        img_draw = self.img.copy()  # 每次把轮廓画到不同的图上 test used
        # img_draw = self.img  # 每次都把轮廓画到同一张图上
        cnt = self.cnts[index]  # 轮廓
        prt('|######################################## 处理轮廓[%d] #######################################|' % index)

        cnts_num, adaptive_cnts = self.adaptive_hsv_counter(cnt)

        if cnts_num != 0:
            prt("真轮廓")
            for cnt in adaptive_cnts:
                popu_num = self.cnts_counting(cnt)
                self.total_popu += popu_num
                # self.draw_number(img_draw, cnt, popu_num)
            self.show_cnt_info(img_draw, adaptive_cnts, index, len(adaptive_cnts))

        else:  # 测试使用
            prt("假轮廓")
            self.show_cnt_info(img_draw, [cnt], index, cnts_num)

        prt('|###################################### 轮廓[{}]有[{}]个细胞 #########################3#########|\n'
            .format(index, len(adaptive_cnts)))

    def adaptive_hsv_counter(self, cnt):
        """自适应阈值计数器：输入局部轮廓图，输出这个轮廓中细胞集落数和处理过的轮廓"""
        # img = self.img.copy()
        config = self.config.copy()

        # 1 小面积直接计数
        # if cv2.contourArea(cnt) < 600:
        #     cv2.drawContours(img, cnt, -1, (255, 255, 0), 1)  # ...把轮廓画出来看看
        #     plt_show(img, figsize=(15, 15))  # test used
        #     return 1
        # 2 所有轮廓均满足条件 全部算数
        rgb_hull = self.rgb_hull(cnt)  # 只有当前轮廓范围的彩色图
        # plt_show(rgb_hull, figsize=(15, 15))
        for i in range(6):
            satisfied, adjusted_cnts, config = self.agent.adjust_hsv_threshold(rgb_hull, config)
            if satisfied:
                return len(adjusted_cnts), adjusted_cnts

        good_adjusted_cnts = []  # 如果没有satisfied，说明存在不满足条件的轮廓，再过滤一下
        for cnt in adjusted_cnts:
            if cv2.contourArea(cnt) >= self.config["good_cell_area"]:
                good_adjusted_cnts.append(cnt)
        # cv2.drawContours(img, good_adjusted_cnts, -1, (255, 255, 0), 1)  # ...把轮廓画出来看看
        # plt_show(img, figsize=(15, 15))  # test used
        return len(good_adjusted_cnts), good_adjusted_cnts

    def cnts_counting(self, cnt):
        cnt_area = cv2.contourArea(cnt)
        prt('轮廓形状分析，面积{}'.format(cnt_area))
        if cnt_area <= 600:  # 一个小的群落
            popu_num = 1
        elif 600 < cnt_area <= 4000:  # 分析连接在一起的轮廓中有几个群落  大轮廓 - 小轮廓 的图
            bin_sub_thresh = self._bin_count_with_hull(cnt)
            popu_num = self.cont_popu_cnts(bin_sub_thresh)
            if popu_num == 0:  # 这个轮廓的计数凹轮廓为0 本身是一个大的圆润的轮廓
                popu_num += 1
        else:
            popu_num = self._bin_count_big_popu(cnt)
        return popu_num

    def cont_popu_cnts(self, img_gray):
        """在生成轮廓的同时，过滤一些面积很小的轮廓
        args:
            img_gray:处理过的二值图像，包含用于计数的轮廓
        ret:
            good_cnts: 计数轮廓
        TODO：这个地方还可以做一些优化
        """
        good_cnts = []
        cnts, _ = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            # print(area)
            if area > self.config["flt_min_area"]:  # 对凹的计数轮廓的面积限定
                good_cnts.append(cnt)
        return len(good_cnts)

    @staticmethod
    def draw_number(img, cnt, num):
        """将轮廓中群落的个数 写在轮廓旁边"""
        cv2.drawContours(img, cnt, -1, (255, 255, 0), 1)  # 将轮廓cnt画到图img_draw上
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.circle(img, (cx, cy), 2, (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(num), (cx + 5, cy - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)

    @staticmethod
    def _draw_index(img, cnt, index):
        """辅助函数：画出轮廓的索引用于后续分析"""
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.circle(img, (cx, cy), 2, (0, 255, 0), 1)
        cv2.putText(img, str(index), (cx + 5, cy + 5),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0), 1)

    @staticmethod
    def show_cnt_info(img, cnts, index, popu_num=1):
        """将轮廓中群落的个数 写在轮廓旁边"""
        h, w, _ = img.shape
        cv2.drawContours(img, cnts, -1, (255, 255, 0), 1)  # 将轮廓cnt画到图img_draw上
        # plt_show(img, figsize=(15, 15))
        try:
            for i, cnt in enumerate(cnts):

                M = cv2.moments(cnt)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(img, (cx, cy), 2, (0, 255, 0), cv2.FILLED)

                if cx + 200 > w:  # 防止字写到图像外面去了
                    cx = cx - 100
                if cy - 80 < 0:
                    cy += 100

                if i < 1:  # index 只写一次
                    cv2.putText(img, 'Idx:' + str(index), (cx + 20, cy - 20),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

                if len(cnts) > 3:  # 单个和多个轮廓采用不同标记方式
                    cv2.putText(img, str(i + 1), (cx + 20, cy + 5),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
                else:
                    cv2.putText(img, 'Num:' + str(popu_num), (cx + 20, cy + 5),
                              cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

        except Exception:
            prt("Missing Info: index[%d]" % index)

        # plt_show(img, figsize=(15, 15))
        plt.imsave("test_imgs/{}.jpg".format(index), img)

    def show_count_result(self):
        cv2.putText(self.img, str(self.total_popu), (50, 80),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 255, 0), 3)
        plt_show(self.img, figsize=(15, 15))

    def rgb_hull(self, cnt):
        """显示原图上某个轮廓凸包下的图像"""
        img_bin = np.zeros((self.h, self.w))
        # hull = cv2.convexHull(cnt, False)
        mask = cv2.drawContours(img_bin, [cnt], -1, 255, cv2.FILLED)
        mask = mask.astype(np.uint8)
        res = cv2.add(self.img, np.zeros(np.shape(self.img), dtype=np.uint8), mask=mask)
        return res

    def bin_contours(self, cnt):
        """在二值图上画出轮廓的填充图"""
        img_bin = np.zeros((self.h, self.w))
        draw_bin = cv2.drawContours(img_bin, [cnt], -1, 255, cv2.FILLED)
        return draw_bin

    def bin_hull(self, cnt):
        """在二值图上画出轮廓凸包的填充图"""
        img_bin = np.zeros((self.h, self.w))
        hull = cv2.convexHull(cnt, False)
        res = cv2.drawContours(img_bin, [hull], -1, 255, cv2.FILLED)
        # plt_show(res, figsize=(15, 15))
        # print(res.shape, res.dtype)
        return res

    def _bin_count_with_hull(self, cnt):
        """使用轮廓外接凸包减去轮廓后剩下的图形作为计数轮廓
        inp：轮廓列表
        out: 凹轮廓二值图
        """

        bin_hull = self.bin_hull(cnt)
        bin_cnt = self.bin_contours(cnt)
        bin_sub = cv2.subtract(bin_hull, bin_cnt)
        bin_sub = bin_sub.astype(np.uint8)  # .astype(np.unit8) np数值类型转化
        _, bin_sub_thresh = cv2.threshold(bin_sub, 50, 255, cv2.THRESH_BINARY)
        # plt_show(bin_hull, figsize=(20,15))
        # plt_show(bin_cnt, figsize=(20,15))
        # plt_show(bin_sub_thresh, figsize=(20, 15))
        return bin_sub_thresh

    def _bin_count_big_popu(self, cnt):
        """这里首要考虑的事情是腐蚀轮廓拉开连在一起的群落"""
        accumulate_popu_num = 0
        bin_cnt = self.bin_contours(cnt)
        # plt_show(bin_cnt, figsize=(20,15))

        kernel = np.ones((3, 3), np.uint8)
        bin_cnt_erode = cv2.erode(bin_cnt, kernel, iterations=4)
        # plt_show(bin_cnt_erode, figsize=(20,15))

        bin_cnt_erode = bin_cnt_erode.astype(np.uint8)
        # _, bin_cnt_erode_thresh = cv2.threshold(bin_cnt_erode, 50, 255, cv2.THRESH_BINARY)
        # plt_show(bin_cnt_erode_thresh, figsize=(20,15))
        #
        # bin_cnt_erode = bin_cnt_erode.astype(np.uint8)
        cnts, _ = cv2.findContours(bin_cnt_erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # print("轮廓个数：", len(cnts))
        for cnt in cnts:
            cnt_area = cv2.contourArea(cnt)
            # print(cnt_area)
            if cnt_area > 6000:  # 如果腐蚀之后轮廓还是很大就直接用面积近似计算
                popu_num = math.ceil(cnt_area / self.config["count_mean_area"])
                # TODO：面积处理  是不是要做递归处理？
                accumulate_popu_num += popu_num
            else:  # 仍然使用凹轮廓计数
                img_bin = self._bin_count_with_hull(cnt)
                # plt_show(img_bin,  figsize=(20,15))
                popu_num = self.cont_popu_cnts(img_bin)
                if popu_num == 0:  # 一个很圆润的轮廓可能没有 凹的计数轮廓
                    popu_num += 1
                accumulate_popu_num += popu_num
        # kernel = np.ones((3,3),np.uint8)
        # bin_cnt_opening = cv2.morphologyEx(bin_cnt_erode, cv2.MORPH_OPEN, kernel)

        # plt_show(bin_cnt_erode_thresh, figsize=(20,15))
        # plt_show(bin_cnt, figsize=(20,15))
        # plt_show(bin_sub_thresh, figsize=(20,15))
        # print('大轮廓中群落的个数为', accumulate_popu_num)
        return accumulate_popu_num


class PopuCntAnalyser:
    """
    分析每个轮廓中一共有几个群落
    以轮廓贯穿处理流程：
        1）自适应颜色阈值 逐个处理用默认阈值筛选出的轮廓
        2）对处理后的颜色轮廓做形状分析
            先画一些辅助图以bin_开头
            没有连接在一起的群落直接计数
            连在一起的群落用“计数轮廓”算细胞个数
    """

    def __init__(self, img_src, config, agent=None, adaptive=True):
        """

        :param counters: cv.findContours); 出入进来初步过滤后的轮廓
        :param img_src: ndarray; 处理过的图像，这里是分割培养皿后的图片
        :param config:
        :param agent:
        :param adaptive:
        """
        self.cnts = agent.cnts  # 传进来所有的轮廓
        self.img = img_src.copy()
        self.config = config
        self.config_temp = config
        self.agent = agent
        self.adaptive = adaptive
        self.total_popu = 0
        self.h, self.w, _ = img_src.shape

    def __getitem__(self, index):
        img_draw = self.img  # 每次都把轮廓画到同一张图上
        cnt = self.cnts[index]  # 轮廓

        cnts_num, adaptive_cnts = self.adaptive_hsv_counter(cnt)

        if cnts_num != 0:
            for cnt in adaptive_cnts:
                popu_num = self.cnts_counting(cnt)
                if popu_num != 0:
                    self.total_popu += popu_num
                    self.draw_number(img_draw, cnt, popu_num)

    def getitem_test(self, index):
        img_draw = self.img.copy()  # 每次把轮廓画到不同的图上 test used
        # img_draw = self.img  # 每次都把轮廓画到同一张图上
        cnt = self.cnts[index]  # 轮廓
        prt('|######################################## 处理轮廓[%d] #######################################|' % index)

        cnts_num, adaptive_cnts = self.adaptive_hsv_counter(cnt)

        if cnts_num != 0:
            prt("真轮廓")
            for cnt in adaptive_cnts:
                popu_num = self.cnts_counting(cnt)
                self.total_popu += popu_num
                # self.draw_number(img_draw, cnt, popu_num)
            self.show_cnt_info(img_draw, adaptive_cnts, index, popu_num)

        else:  # 测试使用
            prt("假轮廓")
            self.show_cnt_info(img_draw, [cnt], index, cnts_num)

        prt('|###################################### 轮廓[{}]有[{}]个细胞 #########################3#########|\n'
            .format(index, len(adaptive_cnts)))

    def adaptive_hsv_counter(self, cnt):
        """自适应阈值计数器：输入局部轮廓图，输出这个轮廓中细胞集落数和处理过的轮廓"""
        # img = self.img.copy()
        config = self.config.copy()

        # 1 小面积直接计数
        # if cv2.contourArea(cnt) < 600:
        #     cv2.drawContours(img, cnt, -1, (255, 255, 0), 1)  # ...把轮廓画出来看看
        #     plt_show(img, figsize=(15, 15))  # test used
        #     return 1
        # 2 所有轮廓均满足条件 全部算数
        rgb_hull = self.rgb_hull(cnt)  # 只有当前轮廓范围的彩色图
        # plt_show(rgb_hull, figsize=(15, 15))
        for i in range(6):
            satisfied, adjusted_cnts, config = self.agent.adjust_hsv_threshold(rgb_hull, config)
            if satisfied:
                return len(adjusted_cnts), adjusted_cnts

        good_adjusted_cnts = []  # 如果没有satisfied，说明存在不满足条件的轮廓，再过滤一下
        for cnt in adjusted_cnts:
            if cv2.contourArea(cnt) >= self.config["good_cell_area"]:
                good_adjusted_cnts.append(cnt)
        # cv2.drawContours(img, good_adjusted_cnts, -1, (255, 255, 0), 1)  # ...把轮廓画出来看看
        # plt_show(img, figsize=(15, 15))  # test used
        return len(good_adjusted_cnts), good_adjusted_cnts

    def cnts_counting(self, cnt):
        (x, y), (w, h), alph = cv2.minAreaRect(cnt)
        if 5 < w < 200 and 5 < h < 200 and 0.25 < w / h < 4:
            cnt_area = cv2.contourArea(cnt)
            prt('轮廓形状分析，面积{}'.format(cnt_area))
            if cnt_area <= 200:  # 一个小的群落
                popu_num = 1
            elif 200 < cnt_area <= 4000:  # 分析连接在一起的轮廓中有几个群落  大轮廓 - 小轮廓 的图
                bin_sub_thresh = self._bin_count_with_hull(cnt)
                popu_num = self.cont_popu_cnts(bin_sub_thresh)
                if popu_num == 0:  # 这个轮廓的计数凹轮廓为0 本身是一个大的圆润的轮廓
                    popu_num += 1
            else:
                popu_num = self._bin_count_big_popu(cnt)
            return popu_num
        else:
            return 0

    def cont_popu_cnts(self, img_gray):
        """在生成轮廓的同时，过滤一些面积很小的轮廓
        args:
            img_gray:处理过的二值图像，包含用于计数的轮廓
        ret:
            good_cnts: 计数轮廓
        TODO：这个地方还可以做一些优化
        """
        good_cnts = []
        cnts, _ = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            # print(area)
            if area > self.config["flt_min_area"]:  # 对凹的计数轮廓的面积限定
                good_cnts.append(cnt)
        return len(good_cnts)

    @staticmethod
    def draw_number(img, cnt, num):
        """将轮廓中群落的个数 写在轮廓旁边"""
        cv2.drawContours(img, cnt, -1, (255, 255, 0), 1)  # 将轮廓cnt画到图img_draw上
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.circle(img, (cx, cy), 2, (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(num), (cx + 5, cy - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)

    @staticmethod
    def _draw_index(img, cnt, index):
        """辅助函数：画出轮廓的索引用于后续分析"""
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.circle(img, (cx, cy), 2, (0, 255, 0), 1)
        cv2.putText(img, str(index), (cx + 5, cy + 5),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0), 1)

    @staticmethod
    def show_cnt_info(img, cnts, index, popu_num):
        """将轮廓中群落的个数 写在轮廓旁边"""
        h, w, _ = img.shape
        cv2.drawContours(img, cnts, -1, (255, 255, 0), 1)  # 将轮廓cnt画到图img_draw上
        # plt_show(img, figsize=(15, 15))
        try:
            for i, cnt in enumerate(cnts):

                M = cv2.moments(cnt)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(img, (cx, cy), 2, (0, 255, 0), cv2.FILLED)

                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)  # int0 等价与int64 将浮点数转换为整数
                cv2.drawContours(img, [box], 0, (0, 0, 255), 1)

                if cx + 200 > w:  # 防止字写到图像外面去了
                    cx = cx - 100
                if cy - 80 < 0:
                    cy += 100

                if i < 1:  # index 只写一次
                    cv2.putText(img, 'Idx:' + str(index), (cx + 20, cy - 20),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

                if len(cnts) > 1 and popu_num != 0:  # 单个和多个轮廓采用不同标记方式
                    cv2.putText(img, str(i + 1), (cx + 20, cy + 5),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
                else:
                    cv2.putText(img, 'Num:' + str(popu_num), (cx + 20, cy + 5),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

        except Exception:
            prt("Missing Info: index[%d]" % index)

        plt_show(img, figsize=(15, 15))

    #         plt.imsave("test_imgs/{}.jpg".format(index), img)

    def show_count_result(self):
        cv2.putText(self.img, str(self.total_popu), (50, 80),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 255, 0), 3)
        plt_show_(self.img, figsize=(15, 15))

    def rgb_hull(self, cnt):
        """显示原图上某个轮廓凸包下的图像"""
        img_bin = np.zeros((self.h, self.w))
        # hull = cv2.convexHull(cnt, False)
        mask = cv2.drawContours(img_bin, [cnt], -1, 255, cv2.FILLED)
        mask = mask.astype(np.uint8)
        res = cv2.add(self.img, np.zeros(np.shape(self.img), dtype=np.uint8), mask=mask)
        return res

    def bin_contours(self, cnt):
        """在二值图上画出轮廓的填充图"""
        img_bin = np.zeros((self.h, self.w))
        draw_bin = cv2.drawContours(img_bin, [cnt], -1, 255, cv2.FILLED)
        return draw_bin

    def bin_hull(self, cnt):
        """在二值图上画出轮廓凸包的填充图"""
        img_bin = np.zeros((self.h, self.w))
        hull = cv2.convexHull(cnt, False)
        res = cv2.drawContours(img_bin, [hull], -1, 255, cv2.FILLED)
        # plt_show(res, figsize=(15, 15))
        # print(res.shape, res.dtype)
        return res

    def _bin_count_with_hull(self, cnt):
        """使用轮廓外接凸包减去轮廓后剩下的图形作为计数轮廓
        inp：轮廓列表
        out: 凹轮廓二值图
        """

        bin_hull = self.bin_hull(cnt)
        bin_cnt = self.bin_contours(cnt)
        bin_sub = cv2.subtract(bin_hull, bin_cnt)
        bin_sub = bin_sub.astype(np.uint8)  # .astype(np.unit8) np数值类型转化
        _, bin_sub_thresh = cv2.threshold(bin_sub, 50, 255, cv2.THRESH_BINARY)
        # plt_show(bin_hull, figsize=(20,15))
        # plt_show(bin_cnt, figsize=(20,15))
        # plt_show(bin_sub_thresh, figsize=(20, 15))
        return bin_sub_thresh

    def _bin_count_big_popu(self, cnt):
        """这里首要考虑的事情是腐蚀轮廓拉开连在一起的群落"""
        accumulate_popu_num = 0
        bin_cnt = self.bin_contours(cnt)
        # plt_show(bin_cnt, figsize=(20,15))

        kernel = np.ones((3, 3), np.uint8)
        bin_cnt_erode = cv2.erode(bin_cnt, kernel, iterations=4)
        # plt_show(bin_cnt_erode, figsize=(20,15))

        bin_cnt_erode = bin_cnt_erode.astype(np.uint8)
        # _, bin_cnt_erode_thresh = cv2.threshold(bin_cnt_erode, 50, 255, cv2.THRESH_BINARY)
        # plt_show(bin_cnt_erode_thresh, figsize=(20,15))
        #
        # bin_cnt_erode = bin_cnt_erode.astype(np.uint8)
        cnts, _ = cv2.findContours(bin_cnt_erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # print("轮廓个数：", len(cnts))
        for cnt in cnts:
            cnt_area = cv2.contourArea(cnt)
            # print(cnt_area)
            if cnt_area > 6000:  # 如果腐蚀之后轮廓还是很大就直接用面积近似计算
                popu_num = math.ceil(cnt_area / self.config["count_mean_area"])
                # TODO：面积处理  是不是要做递归处理？
                accumulate_popu_num += popu_num
            else:  # 仍然使用凹轮廓计数
                img_bin = self._bin_count_with_hull(cnt)
                # plt_show(img_bin,  figsize=(20,15))
                popu_num = self.cont_popu_cnts(img_bin)
                if popu_num == 0:  # 一个很圆润的轮廓可能没有 凹的计数轮廓
                    popu_num += 1
                accumulate_popu_num += popu_num
        # kernel = np.ones((3,3),np.uint8)
        # bin_cnt_opening = cv2.morphologyEx(bin_cnt_erode, cv2.MORPH_OPEN, kernel)

        # plt_show(bin_cnt_erode_thresh, figsize=(20,15))
        # plt_show(bin_cnt, figsize=(20,15))
        # plt_show(bin_sub_thresh, figsize=(20,15))
        # print('大轮廓中群落的个数为', accumulate_popu_num)
        return accumulate_popu_num


# 轮廓数据分析
def cont_df_analysis(contours):
    print('轮廓总个数:', np.array(contours).shape[0])
    cont_arc_length = []  # 轮廓周长
    hull = []  # 轮廓凸包
    hull_arc_length = []  # 凸包周长
    cnt_bounding_rect_w = []
    cnt_bounding_rect_h = []
    for i in range(len(contours)):
        hull.append(cv2.convexHull(contours[i], True))  # 轮廓凸包
        cont_arc_length.append(cv2.arcLength(contours[i], True))  # 轮廓周长
        hull_arc_length.append(cv2.arcLength(hull[i], True))  # 轮廓凸包周长
        x, y, w, h = cv2.boundingRect(contours[i])
        cnt_bounding_rect_w.append(w)
        cnt_bounding_rect_h.append(h)
    cont_area = []
    hull_area = []
    for i in range(len(contours)):
        cont_area.append(cv2.contourArea(contours[i]))
        hull_area.append(cv2.contourArea(hull[i]))

    df_area = pd.DataFrame({
        'area': cont_area,
        'area_hull': hull_area,
        'arc_length': cont_arc_length,
        'arc_length_hull': hull_arc_length,
        'rect_w': cnt_bounding_rect_w,
        'rect_h': cnt_bounding_rect_h,
    })

    df_area['r1'] = df_area['area'] / df_area['arc_length']  # 轮廓面积 / 轮廓周长 = 面积周长比
    df_area['r2'] = df_area['arc_length'] / df_area['arc_length_hull']
    df_area['w/h'] = df_area['rect_w'] / df_area['rect_h']
    df_area['d1'] = df_area['area_hull'] - df_area['area']  # 凸包的面积 - 轮廓的面积

    return df_area
