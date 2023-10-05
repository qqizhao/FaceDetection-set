import argparse

import torch.backends.cudnn as cudnn

from utils import google_utils, torch_utils
import cv2
import os
import shutil
import torch
from utils.datasets import *
from utils.utils import *
from tqdm import tqdm


def detect(opt ,save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32

    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.names if hasattr(model, 'names') else model.modules.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                            file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)

                        return im0, xyxy
                       

def detect_face(choice):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./weights/face.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default=f'{choice}', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='./inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    opt.img_size = check_img_size(opt.img_size)
    # print(opt)

    with torch.no_grad():
        if detect(opt) is not None:
            img, xywh = detect(opt)
            return img, xywh

def crop_image(image, xywh):
    """
    将检测到的人脸区域裁剪出来
    """
    xywh = xywh.numpy() if isinstance(xywh, torch.Tensor) else xywh
    img = image.copy()
    cropped_image = img[int(xywh[1]):int(xywh[3]), int(xywh[0]):int(xywh[2])]
    return cropped_image


if __name__ == '__main__':
    
    image_folder_path = 'I:\Desktop\\face_recognition1\pic'
    root_path = image_folder_path
    
    # 获取所有图像文件
    image_files = []
    for subfolder in os.listdir(root_path):
        subfolder_path = os.path.join(root_path, subfolder)

        if os.path.isdir(subfolder_path):
            image_files += [os.path.join(subfolder_path, filename) for filename in os.listdir(subfolder_path) if filename.lower().endswith(('.jpg', '.png'))]

    # 创建 tqdm 进度条
    progress_bar = tqdm(total=len(image_files), position=0, leave=True)

    # 遍历每个图像文件
    for file_path in image_files:
        # 在这里你可以添加处理进度条的逻辑
        # 更新进度条
        progress_bar.update(1)
        
        if detect_face(file_path) is not None:
            img, xywh = detect_face(file_path)
            cropped_image = crop_image(img, xywh)
            cv2.imwrite(file_path, cropped_image)

    # 关闭进度条
    progress_bar.close()
    print('over!')
 

