# 代码说明

基于YOLOv11的onnx分割。

# 代码结构

```text
README.md  # 自述文件
yolo11m-seg.pt  # pt格式的分割模型。
yolo11m-seg.onnx  # onnx格式的分割模型。
input.jpg  # 一张检测出来的行人小图。
largest_person_mask_0.png  # input.jpg 的分割图像。行人前景的像素值为255，背景的像素值为0。
segmented_largest_person_0.png  # input.jpg 的分割图像。行人前景的像素值为行人像素值，背景的像素值为0。
det_to_seg.py  # 对行人小图进行分割。
person_segmentation.py  # 对整幅图像的所有行人进行实例分割。
```

