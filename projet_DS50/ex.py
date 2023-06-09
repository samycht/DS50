import joblib
import numpy as np
from PIL import Image
from keras.models import load_model
mapping = {
    0:'adenocarcinoma',
    1:'large.cell.carcinoma',
    2:'normal',
    3:'squamous.cell.carcinoma'
}
def predict_image(image):
    model = joblib.load('DT1.dat')
    width, height = 150, 150  # 模型所期望的输入图像的宽度和高度
    # 加载和准备输入图像
    image = image.convert('RGB')
    image = image.resize((width, height))  # 调整图像大小
    image = np.array(image) / 255.0  # 归一化
    print(image.shape)
    # 将图像转换为适当的形状
    image_reshaped = image.reshape((1,150, 150, 3))
    # 进行预测
    predictions = model.predict(image_reshaped)
    
    # 处理预测结果
    predicted_class = np.argmax(predictions[0])  # 获取预测类别的索引
    return mapping[predicted_class]
image = Image.open('000108.png')
res = predict_image(image)
print(res)