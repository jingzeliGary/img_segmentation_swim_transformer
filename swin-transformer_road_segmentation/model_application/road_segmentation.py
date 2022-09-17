'''
使用训练好的swin-transform模型，对道路目标进行分割
1. 初始化模型  model = init_detector(config, checkpoint, device)
2. 获取图片
3. 模型检测   result = inference_detector(model, img)
4. 解析结果  bboxes, labels, segms = analysisResult(result)
5. 绘制结果
'''

import cv2
import numpy as np
import torch
import torch.nn as nn
import mmcv
from mmdet.apis import init_detector,inference_detector,show_result_pyplot

class RoadSeg:
    def __init__(self, config_path,model_path):
        # 初始化模型
        self.config_path = config_path
        self.model_path = model_path
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = init_detector(config=self.config_path,checkpoint=self.model_path,device=self.device)

        # 参数
        self.CLASS = ('arrow', 'car', 'dashed', 'line')
        self.colors = [(255, 0, 255), (0, 255, 255), (0, 255, 0), (0, 0, 255)]
        self.alpha_list = [0.3, 0.5, 0.3, 0.3]


    def analysisResult(self, result):
        '''
        :param result: tuple , 检测结果
        --- bbox_result : list, 记录每个类别(array)的每个目标框的（l,t,r,b,confidence)
        --- segm_result : list, 记录每个类别(array)的基于每个目标框的像素归属

        :return: bboxes, labels, segm 结果
        --- bboxes: numpy, [nums_obj, 5]
        --- labels: numpy, [nums_obj]
        --- segms: numpy, [nums_obj, width_img, height_img]
        '''
        bbox_result, segm_result = result

        bboxes = np.vstack(bbox_result)  # numpy(num_obj, 5)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        segms = mmcv.concat_list(segm_result)
        segms = np.stack(segms, axis=0)

        return bboxes, labels, segms


    def inference(self, video_path, confidence_threshold = 0.5):
        # 摄像头
        cap = cv2.VideoCapture(video_path)
        # width, height
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        while True:
            ret, frame = cap.read()
            # 缩放
            frame = cv2.resize(frame, (int(width / 2), int(height / 2)))
            if ret is False:
                break

            # 模型推断
            result = inference_detector(self.model, frame)   # tuple
            # 解析结果
            bboxes, labels, segms = self.analysisResult(result)

            # 绘制
            for i, bbox in enumerate(bboxes):
                confidence = bbox[-1]
                if confidence > confidence_threshold:
                    l,t,r,b = bbox[:4].astype('int')
                    label = labels[i]
                    segm = segms[i]

                    # 将对应目标框的segm的像素值是True的像素半透明
                    alpha = self.alpha_list[label]
                    color = self.colors[label]
                    frame[segm > 0, 0] = frame[segm > 0, 0] * alpha + color[0] * (1 - alpha)  # B
                    frame[segm > 0, 1] = frame[segm > 0, 1] * alpha + color[1] * (1 - alpha)  # G
                    frame[segm > 0, 2] = frame[segm > 0, 2] * alpha + color[2] * (1 - alpha)  # R

                    # 绘制检测框
                    # cv2.rectangle(frame,(l,t),(r,b),(0,255,0),2)
                    # 绘制类别
                    cv2.putText(frame,str(self.CLASS[labels[i]]),(l,t-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)

            # 显示图片
            cv2.imshow('cap', frame)

            # 关闭条件
            if cv2.waitKey(10) & 0xFF == 27:
                break

        # 释放
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # 实例化
    road_seg = RoadSeg('./weights/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py',
                       './weights/latest.pth')
    # 视频路径
    video_path = './image/lane.MOV'
    # 检测结果
    road_seg.inference(video_path)
