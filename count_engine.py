import cv2
from cell_count_utils import SegCircle, prt
from hsv_agent import HsvImgCntAgent
from contour_analyser import PopuCntAnalyser, cont_df_analysis
from parameter_auto_selector import roi_select, area_select

img_seg = SegCircle()  # 实例化一个培养皿分割器


def cell_population_count(img_path, config):
    img_bgr = cv2.imread(img_path)  # 读入图像并转换格式
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    seg_img_rgb = img_seg(img_rgb)  # 分割出培养皿 抹掉其他部分
    h, s, v = roi_select(seg_img_rgb)

    config["hsv_low"] = [h[0], s[0], v[0]]
    config["hsv_up"] = [h[1], s[1], v[1]]

    hsv_agent = HsvImgCntAgent(seg_img_rgb, config)  # 过滤出轮廓  实例化对象时自动在构造函数中处理了轮廓...

    min_area, max_area = area_select(hsv_agent.cnts)
    config['min_cell_area'] = min_area
    config['max_cell_area'] = max_area
    # print(config)
    popu_analyser = PopuCntAnalyser(seg_img_rgb, config, agent=hsv_agent, adaptive=True)

    for idx in range(len(hsv_agent.cnts)):
        popu_analyser[idx]

    popu_analyser.show_count_result()

    return popu_analyser.total_popu


def cell_population_count_(img_path, config):

    img_bgr = cv2.imread(img_path)  # 读入图像并转换格式
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    seg_img_rgb = img_seg(img_rgb)  # 分割出培养皿 抹掉其他部分

    hsv_agent = HsvImgCntAgent(seg_img_rgb, config)  # 过滤出轮廓  实例化对象时自动在构造函数中处理了轮廓...

    popu_analyser = PopuCntAnalyser(seg_img_rgb, config, agent=hsv_agent, adaptive=True)

    for idx in range(len(hsv_agent.cnts)):
        popu_analyser[idx]

    popu_analyser.show_count_result()

    return popu_analyser.total_popu


def cell_population_count_test(img_path, config):

    img_bgr = cv2.imread(img_path)  # 读入图像并转换格式
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    seg_img_rgb = img_seg(img_rgb)  # 分割出培养皿 抹掉其他部分

    hsv_agent = HsvImgCntAgent(seg_img_rgb)  # 过滤出轮廓  实例化对象时自动在构造函数中处理了轮廓...

    df_cnts = cont_df_analysis(hsv_agent.cnts)
    print(df_cnts.iloc[[30, 42, 102, 114]])

    # popu_analyser = PopuCntAnalyser(seg_img_rgb, config, agent=hsv_agent, adaptive=True)

    # index = 0
    # i = len(hsv_agent.cnts)
    # for idx in range(index, index + i):
    #     popu_analyser.getitem_test(idx)
    #
    # popu_analyser.show_count_result()
    # prt('检测总轮廓数', popu_analyser.total_popu)
    # return popu_analyser.total_popu


if __name__ == "__main__":
    # 用于自适应调整的参数配置变量
    # self_adaptive_config = {
    #     "hsv_low": [100, 20, 50],
    #     "hsv_up": [200, 250, 150],
    #     "flt_min_area": 8,
    #     "count_mean_area": 1400,
    #     "min_cell_area": 70,  # 120
    #     "max_cell_area": 5000,  # 2500
    #     "adjust_ratio_low": 0.1,
    #     "adjust_ratio_up": 0.0005,
    #     "good_cell_area": 80  # 80 40
    # }
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
        "adjust_ratio_low": [0.4, 0.2, 0.4],   # 0.1,  # 0.01
        "adjust_ratio_up":  [0.04, 0.02, 0.04],  # 0.001
        "good_cell_area": 40  # 80 40
    }

    # img_file = "data/cells1/308.jpg"
    # img_file = "data/cells1/187.jpg"
    # img_file = "data/cells2/355.jpg"
    img_file = "data/cells2/177.jpg"

    cell_population_count(img_file, self_adaptive_config_cell2)
