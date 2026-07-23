"""实时检测：摄像头 or 屏幕区域 — 螺丝 vs 螺母"""
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.mobilenetv2 import mobilenet_v2
import cv2
import numpy as np
from PIL import Image, ImageGrab
import argparse

MODEL_PATH = "output/best_model.pth"
IMG_SIZE = 224
LABELS = ["screw (螺丝)", "washer (螺母)"]
CONFIDENCE_THRESHOLD = 0.80
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def build_model(num_classes=2):
    model = mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

def select_screen_roi():
    """截取全屏让用户框选检测区域"""
    print("⏳ 正在截取全屏，请用鼠标框选检测区域...")
    screenshot = ImageGrab.grab()
    img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    scale = min(1200 / w, 900 / h, 1.0)
    if scale < 1.0:
        disp_w, disp_h = int(w * scale), int(h * scale)
        display = cv2.resize(img, (disp_w, disp_h))
    else:
        display = img.copy()
        scale = 1.0

    cv2.putText(display, "拖动鼠标框选区域, SPACE/ENTER确认, ESC重选",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("选择检测区域", display)
    roi = cv2.selectROI("选择检测区域", display, False)
    cv2.destroyWindow("选择检测区域")

    x, y, rw, rh = [int(v / scale) for v in roi]
    if rw < 10 or rh < 10:
        print("❌ 区域太小，使用默认居中 400x400")
        x, y = w // 2 - 200, h // 2 - 200
        rw, rh = 400, 400
    print(f"📌 已选区域: ({x}, {y})  {rw}x{rh}")
    return x, y, rw, rh

def run_inference(frame, model):
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(prob, dim=1).item()
        confidence = prob[0][pred_idx].item()
    if confidence < CONFIDENCE_THRESHOLD:
        return None, confidence
    return pred_idx, confidence

def draw_ui(frame, pred_idx, confidence, mode="camera"):
    result = cv2.resize(frame, (480, 360)) if mode == "screen" else frame.copy()
    h, w = result.shape[:2]

    if pred_idx is None:
        label, color, box_color = "未识别", (128, 128, 128), (128, 128, 128)
    else:
        label = LABELS[pred_idx]
        color = box_color = (0, 255, 0) if pred_idx == 0 else (255, 0, 0)

    cv2.rectangle(result, (0, 0), (w, 10), color, -1)
    cv2.putText(result, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(result, f"Conf: {confidence:.2%}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.rectangle(result, (10, 120), (350, 170), box_color, 2)
    txt = f">> {label} <<" if pred_idx is not None else ">> ??? <<"
    cv2.putText(result, txt, (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
    cv2.putText(result, "[屏幕检测]" if mode == "screen" else "[摄像头]",
                (w - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return result

def main():
    parser = argparse.ArgumentParser(description="螺丝/螺母 实时检测")
    parser.add_argument("-c", "--camera", type=int, default=None, help="摄像头索引")
    parser.add_argument("-s", "--screen", action="store_true", help="截屏模式: 框选屏幕区域检测")
    args = parser.parse_args()

    print(f"使用设备: {DEVICE}")
    print("加载模型中...")

    model = build_model(2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    if args.screen:
        # ====== 截屏模式 ======
        x, y, rw, rh = select_screen_roi()
        print(f"\n🟢 屏幕检测已启动，按 Q 退出，按 S 截图")
        print(f"检测区域: ({x},{y}) ~ ({x+rw},{y+rh})")

        while True:
            screenshot = ImageGrab.grab(bbox=(x, y, x + rw, y + rh))
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            pred_idx, confidence = run_inference(frame, model)
            result = draw_ui(frame, pred_idx, confidence, mode="screen")
            cv2.imshow("螺丝/螺母 屏幕检测 (Q退出)", result)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite("screenshot.jpg", frame)
                print("📸 截图已保存: screenshot.jpg")
    else:
        # ====== 摄像头模式 ======
        camera_idx = args.camera if args.camera is not None else 0
        cap = None
        for idx in range(camera_idx, 5):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                print(f"📷 使用摄像头: 索引 {idx}")
                camera_idx = idx
                break
            cap.release()
        if cap is None or not cap.isOpened():
            print("❌ 无法打开摄像头！")
            return
        print("\n🟢 摄像头已打开，按 Q 退出，按 S 截图")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            pred_idx, confidence = run_inference(frame, model)
            result = draw_ui(frame, pred_idx, confidence, mode="camera")
            cv2.imshow("螺丝/螺母 实时检测 (Q退出)", result)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite("screenshot.jpg", frame)
                print("📸 截图已保存: screenshot.jpg")
        cap.release()

    cv2.destroyAllWindows()
    print("👋 已退出")

if __name__ == "__main__":
    main()
