"""测试两张图片的推理结果"""
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms

t = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

for mdl_path in ['output/model_int8.onnx', 'output/model_fixed.onnx']:
    sess = ort.InferenceSession(mdl_path)
    input_name = sess.get_inputs()[0].name
    tag = mdl_path.split('/')[-1]
    print(f'--- {tag} ---')
    for img_name in ['test_images/t1.jpg', 'test_images/t2.jpg']:
        img = Image.open(img_name).convert('RGB')
        x = t(img).unsqueeze(0).numpy()
        out = sess.run(None, {input_name: x})[0][0]
        print(f'  {img_name}:  [{out[0]:.4f}, {out[1]:.4f}]')
