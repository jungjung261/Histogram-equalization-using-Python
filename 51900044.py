import numpy as np
import cv2
from matplotlib import pyplot as plt


def Histogram(img):
    # Dùng thư viện CV2 để mở ảnh
    img = cv2.imread('Source_HistogramWiki.png')

    # Tách các luồng màu (số lượng điểm ảnh)
    row, col = np.histogram(img.ravel(), 256, [0, 256])

    # Tính tổng phân phối các số lượng điểm ảnh
    cdf = np.cumsum(row)

    # Thay thế các điểm ảnh có giá trị bằng 0 bằng cách thay thế các điểm ảnh có giá trị trung bình
    # Dùng ma.masked_equal để che các điểm ảnh có giá trị bằng 0
    pixel = np.ma.masked_equal(cdf, 0)

    # Tính giá trị trung bình
    pixel = (pixel - pixel.min()) * 255 / (pixel.max() - pixel.min())  # giới hạn L - 1
    
    # Thay thế giá trị điểm ảnh mới bằng hàm ma.filled
    temp = np.ma.filled(pixel, 0).astype('uint8')  # unit8 có số lượng màu là 256
    result = temp[img]
    # Ghi kết quả vào ảnh mới
    cv2.imwrite('Completed_Histogram.png', result)

    # Vẽ biểu đồ Histogram
    cdf_histogram = cdf * float(row.max()) / cdf.max()
    plt.plot(cdf_histogram, color='g')  # số lượng điểm ảnh - màu xanh
    plt.hist(img.ravel(), 256, [0, 256], color='r')# cường độ ánh sáng - màu đỏ
    plt.xlim([0, 256])
    plt.show()


img = cv2.imread('Source_HistogramWiki.png')
Histogram(img)
