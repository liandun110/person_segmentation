"""
本文件中，输入图像具备如下特征：
（1）是仅包含一个行人的小图像
输出结果应具备如下特征：
（1）input.jpg 的分割图像。行人前景的像素值为255，背景的像素值为0。
（2）仅关注“person”的分割结果
"""
import onnxruntime as ort
import cv2
import torch
import numpy as np
from person_segmentation import preprocess, postprocess

def main():
    session = ort.InferenceSession("yolo11m-seg.onnx")
    img_path = 'input.jpg'

    # 读取图像
    image = cv2.imread(img_path)
    img = preprocess([image])

    # 转为 NumPy 数组以进行 ONNX 推理
    img = img.cpu().numpy()
    input_name = session.get_inputs()[0].name

    # ONNX 推理
    result = session.run(None, {input_name: img})
    preds = [torch.tensor(result[0]), torch.tensor(result[1])]

    # 后处理
    results = postprocess(preds, img, [image], [img_path])

    # # 遍历结果，只保留面积最大的行人掩码
    # for idx, result in enumerate(results):
    #     masks = result['masks']  # 获取分割掩码
    #     orig_img = result['orig_img']  # 原始图像
    #
    #     # 获取原始图像尺寸
    #     orig_h, orig_w = orig_img.shape[:2]
    #
    #     # 找到面积最大的掩码
    #     max_area = 0
    #     largest_mask = None
    #     for mask in masks:
    #         mask_resized = mask.cpu().numpy().astype(np.uint8)  # 转为 NumPy
    #         mask_resized = cv2.resize(mask_resized, (img.shape[3], img.shape[2]), interpolation=cv2.INTER_NEAREST)
    #
    #         # 计算 letterbox 添加的边距
    #         padded_h, padded_w = img.shape[2:]  # 经过 letterbox 后的尺寸
    #         ratio = min(padded_w / orig_w, padded_h / orig_h)  # 缩放比例
    #         dw, dh = (padded_w - ratio * orig_w) / 2, (padded_h - ratio * orig_h) / 2  # 计算边距
    #
    #         # 对掩码进行位置调整
    #         mask_resized = mask_resized[int(dh):int(dh + orig_h * ratio), int(dw):int(dw + orig_w * ratio)]
    #         mask_resized = cv2.resize(mask_resized, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)  # 调整为原始大小
    #
    #         # 计算掩码面积
    #         area = np.sum(mask_resized > 0)
    #         if area > max_area:
    #             max_area = area
    #             largest_mask = mask_resized
    #
    #     # 如果找到最大掩码，保存结果
    #     if largest_mask is not None:
    #         # 确保掩码二值化并设置前景为 255，背景为 0
    #         mask_binary = (largest_mask > 0).astype(np.uint8) * 255
    #
    #         # 保存掩码为单通道图像
    #         mask_filename = f"largest_person_mask_{idx}.png"
    #         cv2.imwrite(mask_filename, mask_binary)
    #     else:
    #         # 如果没有找到最大掩码，保存纯黑图像
    #         black_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    #         mask_filename = f"largest_person_mask_{idx}.png"
    #         cv2.imwrite(mask_filename, black_mask)
    # # 遍历结果，只保存面积最大的行人分割结果
    # for idx, result in enumerate(results):
    #     masks = result['masks']  # 获取分割掩码
    #     orig_img = result['orig_img']  # 原始图像
    #
    #     # 获取原始图像尺寸
    #     orig_h, orig_w = orig_img.shape[:2]
    #
    #     # 找到面积最大的掩码
    #     max_area = 0
    #     largest_mask = None
    #     for mask in masks:
    #         mask_resized = mask.cpu().numpy().astype(np.uint8)  # 转为 NumPy
    #         mask_resized = cv2.resize(mask_resized, (img.shape[3], img.shape[2]), interpolation=cv2.INTER_NEAREST)
    #
    #         # 计算 letterbox 添加的边距
    #         padded_h, padded_w = img.shape[2:]  # 经过 letterbox 后的尺寸
    #         ratio = min(padded_w / orig_w, padded_h / orig_h)  # 缩放比例
    #         dw, dh = (padded_w - ratio * orig_w) / 2, (padded_h - ratio * orig_h) / 2  # 计算边距
    #
    #         # 对掩码进行位置调整
    #         mask_resized = mask_resized[int(dh):int(dh + orig_h * ratio), int(dw):int(dw + orig_w * ratio)]
    #         mask_resized = cv2.resize(mask_resized, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)  # 调整为原始大小
    #
    #         # 计算掩码面积
    #         area = np.sum(mask_resized > 0)
    #         if area > max_area:
    #             max_area = area
    #             largest_mask = mask_resized
    #
    #     # 如果找到最大掩码，保存分割结果
    #     if largest_mask is not None:
    #         # 确保掩码二值化
    #         mask_binary = (largest_mask > 0).astype(np.uint8)
    #
    #         # 创建分割结果图像（背景变黑）
    #         segmented_img = orig_img.copy()
    #         segmented_img[mask_binary == 0] = 0  # 将掩码外的像素置为黑色
    #
    #         # 保存分割结果
    #         result_filename = f"segmented_largest_person_{idx}.png"
    #         cv2.imwrite(result_filename, segmented_img)
    #     else:
    #         # 如果没有找到最大掩码，保存原始图像的黑色版本
    #         black_img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    #         result_filename = f"segmented_largest_person_{idx}.png"
    #         cv2.imwrite(result_filename, black_img)

    # 遍历结果，保存面积最大的行人掩码和分割结果
    for idx, result in enumerate(results):
        masks = result['masks']  # 获取分割掩码
        orig_img = result['orig_img']  # 原始图像

        # 获取原始图像尺寸
        orig_h, orig_w = orig_img.shape[:2]

        # 找到面积最大的掩码
        max_area = 0
        largest_mask = None
        for mask in masks:
            mask_resized = mask.cpu().numpy().astype(np.uint8)  # 转为 NumPy
            mask_resized = cv2.resize(mask_resized, (img.shape[3], img.shape[2]), interpolation=cv2.INTER_NEAREST)

            # 计算 letterbox 添加的边距
            padded_h, padded_w = img.shape[2:]  # 经过 letterbox 后的尺寸
            ratio = min(padded_w / orig_w, padded_h / orig_h)  # 缩放比例
            dw, dh = (padded_w - ratio * orig_w) / 2, (padded_h - ratio * orig_h) / 2  # 计算边距

            # 对掩码进行位置调整
            mask_resized = mask_resized[int(dh):int(dh + orig_h * ratio), int(dw):int(dw + orig_w * ratio)]
            mask_resized = cv2.resize(mask_resized, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)  # 调整为原始大小

            # 计算掩码面积
            area = np.sum(mask_resized > 0)
            if area > max_area:
                max_area = area
                largest_mask = mask_resized

        # 如果找到最大掩码
        if largest_mask is not None:
            # 确保掩码二值化
            mask_binary = (largest_mask > 0).astype(np.uint8)

            # 保存掩码为单通道图像
            mask_filename = f"largest_person_mask_{idx}.png"
            cv2.imwrite(mask_filename, mask_binary * 255)  # 保存二值图像

            # 创建分割结果图像（背景变黑）
            segmented_img = orig_img.copy()
            segmented_img[mask_binary == 0] = 0  # 将掩码外的像素置为黑色

            # 保存分割结果
            result_filename = f"segmented_largest_person_{idx}.png"
            cv2.imwrite(result_filename, segmented_img)
        else:
            # 如果没有找到最大掩码，保存纯黑图像作为掩码和分割结果
            black_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            black_img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

            # 保存纯黑掩码
            mask_filename = f"largest_person_mask_{idx}.png"
            cv2.imwrite(mask_filename, black_mask)

            # 保存纯黑分割结果
            result_filename = f"segmented_largest_person_{idx}.png"
            cv2.imwrite(result_filename, black_img)


if __name__ == '__main__':
    main()
