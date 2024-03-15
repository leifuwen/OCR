import cv2
import pytesseract
import numpy as np

# 读取图像
image = cv2.imread('answer_sheet.jpg')

# 检查图像是否正确读取
if image is None:
    print("Error: Image cannot be read.")
    exit()
# 转换图像为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 使用高斯滤波去噪
denoised = cv2.GaussianBlur(gray, (5, 5), 0)
# 二值化
_, binary = cv2.threshold(denoised, 128, 255, cv2.THRESH_BINARY_INV)
# 检测图像中的文本区域
edges = cv2.Canny(binary, 50, 150, apertureSize=3)

# 找到边缘的最大角度
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
if lines is not None:
    max_angle = 0
    for rho, theta in lines[:, 0]:
        angle = np.degrees(theta)
        if angle > max_angle:
            max_angle = angle

    # 计算图像倾斜的角度
    angle_to_rotate = max_angle if max_angle < 45 else 90 - max_angle

    # 获取旋转矩阵
    center = (binary.shape[1] / 2, binary.shape[0] / 2)
    matrix = cv2.getRotationMatrix2D(center, angle_to_rotate, 1)

    # 进行图像倾斜校正
    corrected_image = cv2.warpAffine(binary, matrix, (binary.shape[1], binary.shape[0]))

# 使用Tesseract进行OCR
text = pytesseract.image_to_string(binary, config='--oem 3 --psm 6',lang='ans')
# 转义非法字符
text = text.encode('gbk', 'ignore').decode('gbk')
print(text)
# 创建一个副本图像，用于绘制边界框
image_with_text = np.copy(binary)
# 获取文本的边界框
boxes = pytesseract.image_to_data(binary, output_type=pytesseract.Output.DICT)
# 遍历每个字符和其边界框
for i in range(len(boxes['level'])):
    if int(boxes['level'][i]) == 1:
        # 获取字符的边界框
        (x, y, w, h) = (boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i])
        # 在副本图像上绘制边界框
        cv2.rectangle(image_with_text, (x, y), (x + w, y + h), (0, 255, 0), 2)
# 调整图像尺寸以适应显示窗口
image_resized = cv2.resize(image_with_text, (600, 400))  # 调整图像宽度为800像素，高度为600像素

# 显示图像
cv2.imshow('Image with text boxes', image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()