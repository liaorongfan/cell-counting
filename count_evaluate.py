"""
# -*- coding: utf-8 -*-
# CreateBy: liaorongfan
# CreateAT: 2020/9/16
"""
import os
import numpy as np
from count_engine import cell_population_count
from cell_count_utils import count_metric, count_record


self_adaptive_config_cell = {
    "initial_filtered_cnts_area": 20,
    "hsv_low": [100, 20, 50],
    "hsv_up": [200, 250, 150],
    "flt_min_area": 20,
    "count_mean_area": 1400,
    "min_cell_area": 100,  # 120
    "max_cell_area": 5000,  # 2500
    "adjust_ratio_low": [0.4, 0.1, 0.4],
    "adjust_ratio_up": [0.04, 0.05, 0.04],
    "good_cell_area": 40  # 80 40
}


self_adaptive_config_cell2 = {
    "initial_filtered_cnts_area": 20,
    "hsv_low": [100, 20, 50],
    "hsv_up": [200, 250, 200],
    "flt_min_area": 10,
    "count_mean_area": 1400,
    "min_cell_area": 100,  # 120
    "max_cell_area": 4000,  # 3000
    "adjust_ratio_low": 0.02,  # 0.01
    "adjust_ratio_up": 0.0002,  # 0.0002
    "good_cell_area": 40  # 80 40
}


def evaluate(dir_list, config):
    errors = []
    for direct in dir_list:
        img_files2 = os.listdir(direct)
        images = [os.path.join(direct, item) for item in img_files2 if item.endswith("jpg")]
        count_num = []
        for img in images:
            cell_num = cell_population_count(img, config)
            count_num.append(cell_num)
            print(img, ':', cell_num)

        labeled_num = [item.split(".")[0] for item in img_files2 if item.endswith("jpg")]
        labeled_num = list(map(int, labeled_num))
        error = count_metric(count_num, labeled_num)
        print('误差：{} | 均值：{:.2f}'.format(error, np.array(error).mean()))
        # 记录
        count_record(config, direct, labeled_num, count_num, error)
        errors.extend(error)
    print('总体平均误差:{:.3f}'.format(np.array(errors).mean()))


if __name__ == "__main__":
    dirs = ["./data/cells1", "./data/cells2"]
    # dirs = ["./data/cells1"]
    evaluate(dirs, self_adaptive_config_cell)
