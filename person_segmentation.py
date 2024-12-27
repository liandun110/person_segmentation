import onnxruntime as ort
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
device='cuda:0'
def letterbox(img,new_shape=(640, 640), center=True):
    labels = {}
    shape = img.shape[:2]  # current shape [height, width]
    new_shape = labels.pop("rect_shape", new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if center:
        dw /= 2  # divide padding into 2 sides
        dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)) if center else 0, int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)) if center else 0, int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )  # add border
    return img

def pre_transform(im):
    return [letterbox(img=x) for x in im]

def preprocess(im):
    im = np.stack(pre_transform(im))
    im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im)

    im = im.to(device)
    im = im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    return im
def xywh2xyxy(x):
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y

def clip_boxes(boxes, shape):  # faster individually (WARNING: inplace .clamp_() Apple MPS bug)
    boxes[..., 0] = boxes[..., 0].clamp(0, shape[1])  # x1
    boxes[..., 1] = boxes[..., 1].clamp(0, shape[0])  # y1
    boxes[..., 2] = boxes[..., 2].clamp(0, shape[1])  # x2
    boxes[..., 3] = boxes[..., 3].clamp(0, shape[0])  # y2
    return boxes

def scale_boxes(img1_shape, boxes, img0_shape):

    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (
        round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
        round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
    )  # wh padding

    boxes[..., 0] -= pad[0]  # x padding
    boxes[..., 1] -= pad[1]  # y padding
    boxes[..., 2] -= pad[0]  # x padding
    boxes[..., 3] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)

def crop_mask(masks, boxes):
    _, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    max_det=300,
    nc=80,  # number of classes (optional)
    max_nms=30000,
    max_wh=7680,
):
    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4  # number of masks
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)

    prediction[..., :4] = xywh2xyxy(prediction[..., :4])

    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inferencet
        x = x[xc[xi]]  # confidence

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == classes).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        # c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # scores = x[:, 4]  # scores
        #
        # boxes = x[:, :4] + c  # boxes (offset by class)
        # i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        #
        # i = i[:max_det]  # limit detections
        # output[xi] = x[i]

        boxes = x[:, :4].cpu().numpy()  # boxes (xyxy format)
        scores = x[:, 4].cpu().numpy()  # scores

        # 确保 boxes 是列表的列表
        boxes = boxes.tolist()

        # 确保 scores 是一维列表
        scores = scores.flatten().tolist()

        # 使用 cv2.dnn.NMSBoxes 进行 NMS
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=conf_thres, nms_threshold=iou_thres)

        # 处理 NMS 输出
        if len(indices) > max_det:
            indices = indices[:max_det]  # limit detections

        output[xi] = x[torch.tensor(indices)]  # 选择保留的框

    return output

def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)  # CHW
    width_ratio = mw / iw
    height_ratio = mh / ih

    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= width_ratio
    downsampled_bboxes[:, 2] *= width_ratio
    downsampled_bboxes[:, 3] *= height_ratio
    downsampled_bboxes[:, 1] *= height_ratio

    masks = crop_mask(masks, downsampled_bboxes)  # CHW

    if upsample:
        masks = F.interpolate(masks[None], shape, mode="bilinear", align_corners=False)[0]  # CHW
    return masks.gt_(0.0)

def postprocess(preds, img, orig_imgs,img_path_list):
    p = non_max_suppression(
        preds[0],
        conf_thres=0.25,
        iou_thres=0.7,
        agnostic=False,
        max_det=300,
        classes=None,
    )


    batch_list = img_path_list

    results = []
    proto = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]

    for i, (pred, orig_img, img_path) in enumerate(zip(p, orig_imgs, batch_list)):
        pred = pred[pred[:, 5] == 0]  # 过滤掉非人的类别。若注释掉这句话，则可显示所有类别。

        if len(pred) == 0:  # 如果没有检测到“人”，跳过当前图像
            continue
        masks = process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
        pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)

        boxes = {
            'xyxy': pred[:, :4],  # Bounding box coordinates (xyxy)
            'conf': pred[:, 4],  # Confidence scores
            'cls': pred[:, 5],  # Class labels
        }

        # 将结果保存为字典
        result = {
            'boxes': boxes,  # Bounding box信息
            'keypoints': None,  # 如果需要关键点检测，可以在这里填充
            'masks': masks , # 如果需要分割结果，可以在这里填充
            'names': {0: 'person',5:'bus'},  # 类别名称字典
            'obb': None,  # 如果有有向边界框（OBB）信息，可以在这里填充
            'orig_img': orig_img,  # 原始图像
            'orig_shape': orig_img.shape,  # 原始图像的尺寸
            'path': img_path,  # 图像路径
            'probs': boxes['conf'],  # 分类概率，默认为 None
            'save_dir': None,  # 保存路径，如果需要保存图像或结果
            'speed': {'preprocess': None, 'inference': None, 'postprocess': None}  # 处理速度信息
        }

        # 将字典添加到结果列表
        results.append(result)
    return results

def main():
    session = ort.InferenceSession("yolo11m-seg.onnx")

    img_path='bus.jpg'
    image = cv2.imread(img_path)
    img=preprocess([image])

    img = img.cpu().numpy()
    input_name = session.get_inputs()[0].name

    result = session.run(None, {input_name: img})
    preds = [torch.tensor(result[0]), torch.tensor(result[1])]

    # 后处理并可视化
    results=postprocess(preds, img, [image],[img_path])

    for result in results:
        boxes = result['boxes']  # 获取 boxes 对象
        probs = result['probs']  # 获取分类概率
        masks = result['masks']  # 获取分割掩码

        # 获取原始图像（假设 orig_img 是原始图像）
        orig_img = result['orig_img']

        # 获取边界框坐标、置信度和类别
        xyxy = boxes['xyxy'].numpy()  # 转换为 NumPy 数组，便于处理
        conf = boxes['conf'].numpy()  # 置信度
        cls = boxes['cls'].numpy()  # 类别标签

        # 获取类别名称
        class_names = result['names']

        # 遍历每个框并进行绘制
        for i, (box, confidence, class_id, mask) in enumerate(zip(xyxy, conf, cls, masks)):
            # 获取边界框的坐标
            x1, y1, x2, y2 = map(int, box)

            # 获取分类标签和置信度
            class_name = class_names[int(class_id)]  # 类别名称
            label = f"{class_name} {confidence:.2f}"  # 标签文本，显示类别和置信度

            # 绘制边界框
            cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绘制绿色矩形框

            # 在边界框上方显示文本
            cv2.putText(orig_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 调整掩码位置以匹配原始图像
            orig_h, orig_w = orig_img.shape[:2]
            mask_resized = mask.cpu().numpy().astype(np.uint8)  # 转为 NumPy
            mask_resized = cv2.resize(mask_resized, (img.shape[3], img.shape[2]), interpolation=cv2.INTER_NEAREST)

            # 计算 letterbox 添加的边距
            padded_h, padded_w = img.shape[2:]  # 经过 letterbox 后的尺寸
            ratio = min(padded_w / orig_w, padded_h / orig_h)  # 缩放比例
            dw, dh = (padded_w - ratio * orig_w) / 2, (padded_h - ratio * orig_h) / 2  # 计算边距

            # 对掩码进行位置调整
            mask_resized = mask_resized[int(dh):int(dh + orig_h * ratio), int(dw):int(dw + orig_w * ratio)]
            mask_resized = cv2.resize(mask_resized, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)  # 调整为原始大小

            # 确保掩码二值化
            mask_resized = (mask_resized > 0).astype(np.uint8)

            # 创建绿色掩膜
            green_mask = np.zeros_like(orig_img, dtype=np.uint8)
            green_mask[:, :, 1] = 255

            # 将掩膜叠加到原图上
            orig_img = np.where(mask_resized[:, :, None] == 1, green_mask, orig_img)

        # 显示图像
        plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))  # 将 BGR 转换为 RGB
        plt.axis('off')  # 关闭坐标轴
        plt.show()  # 显示图像



if __name__=="__main__":
    main()
