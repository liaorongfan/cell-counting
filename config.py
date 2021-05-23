"""
# -*- coding: utf-8 -*-
# CreateBy: liaorongfan
# CreateAT: 2020/9/17
"""
self_adaptive_config_cell1 = {
    "initial_filtered_cnts_area": 20,
    "hsv_low": [100, 20, 50],
    "hsv_up": [200, 250, 150],
    "flt_min_area": 20,
    "count_mean_area": 1400,
    "min_cell_area": 100,  # 120
    "max_cell_area": 3000,  # 2500
    "adjust_ratio_low": 0.1,
    "adjust_ratio_up": 0.0005,
    "good_cell_area": 30  # 80 40
}

self_adaptive_config_cell2 = {
    "initial_filtered_cnts_area": 20,
    "hsv_low": [100, 20, 50],
    "hsv_up": [200, 250, 150],
    "flt_min_area": 20,
    "count_mean_area": 1400,
    "min_cell_area": 100,  # 120
    "max_cell_area": 3000,  # 2000
    "adjust_ratio_low": 0.01,  # 0.01
    "adjust_ratio_up": 0.001,  # 0.001
    "good_cell_area": 40  # 80 40
}
