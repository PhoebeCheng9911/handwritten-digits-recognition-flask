
import base64

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from cnn_mnist import CnnNet
from MLP_DNN import NeuralNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def findBorderContours(img, maxArea=50):

    converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    img = accessBinary(converted_img)


    contours,_ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    borders = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > maxArea:
            border = [(x, y), (x + w, y + h)]
            borders.append(border)
    return borders

def accessPiexl(img):
    height = img.shape[0]
    width = img.shape[1]
    for i in range(height):
       for j in range(width):
           img[i][j] = 255 - img[i][j]
    return img


def denoise_demo(src):
    dst = cv2.fastNlMeansDenoisingColored(src, None, 15, 15, 7, 21)

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gret = cv2.fastNlMeansDenoising(gray, None, 15, 8, 25)
    return gret


def salt_pepper_noise(src):
    dst = cv2.medianBlur(src, 5)
    return dst


def accessBinary(img, threshold=128):
    img = accessPiexl(img)
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    _, img = cv2.threshold(img, threshold, 0, cv2.THRESH_TOZERO)
    return img

def transMNIST(img, borders, size=(28, 28)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgData = np.zeros((len(borders), size[0], size[0], 1), dtype='uint8')
    img = accessBinary(img)
    for i, border in enumerate(borders):
        borderImg = img[border[0][1]:border[1][1], border[0][0]:border[1][0]]
        extendPiexl = (max(borderImg.shape) - min(borderImg.shape)) // 2
        targetImg = cv2.copyMakeBorder(borderImg, 7, 7, extendPiexl + 7, extendPiexl + 7, cv2.BORDER_CONSTANT)

        targetImg = cv2.resize(targetImg, size)
        targetImg = np.expand_dims(targetImg, axis=-1)
        imgData[i] = targetImg
    return imgData

def predict(my_mnist_model, imgData):

    my_mnist_model.eval()
    with torch.no_grad():

        img = imgData.astype('float32') / 255


        for i in range(len(img)):
            plt.figure("Image")  
            plt.imshow(img[i],cmap='gray')
            plt.axis('on')  
            plt.title('hand write digit rec')  
            plt.show()
            

        img = torch.tensor(img.reshape(-1, 28 * 28))
        results = my_mnist_model(img).numpy()
        result_number = []
        for result in results:
            result_number.append(np.argmax(result))

        return result_number

def mnist_digit_rec(str_img):
    data = {'numbers': [],'marked_img':''}
    str_img = str_img.strip().replace("data:image/png;base64,", '').replace("data:image/jpg;base64,", '')
    print(str_img)
    img_data = base64.b64decode(str_img)
    nparr = np.fromstring(img_data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    data['numbers'],data['marked_img']=ndarrayImg2Numbers(img_np);
    return data

def ndarrayImg2Numbers(ndarray_img):
    input_size = 784
    hidden_size = 500
    output_size = 10

    my_mnist_model = NeuralNet(input_size,hidden_size,output_size)
    my_mnist_model.load_state_dict(torch.load('model.ckpt'))#'model.ckpt'

    borders = findBorderContours(ndarray_img)
    imgData = transMNIST(ndarray_img, borders)
    results = predict(my_mnist_model, imgData)
    marked_img = showResults(ndarray_img, 'post_request_img.jpg', borders, results)
    return results,marked_img

def showResults(img,save_path, borders, results=None):

    print(img.shape)
    for i, border in enumerate(borders):
        cv2.rectangle(img, border[0], border[1], (0, 0, 255))
        if results:
            cv2.putText(img, str(results[i]), border[0], cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
        # cv2.circle(img, border[0], 1, (0, 255, 0), 0)

    #cv2.imshow('test', img)
    cv2.imwrite(save_path, img)
    #cv2.waitKey(0)
    plt.figure("Image") 
    plt.imshow(img)
    plt.axis('on')  
    plt.title('hand write digit rec')  
    plt.show()


    encoded = cv2.imencode('.jpg',img)
    #image_code = str(base64.b64encode(encoded[1]))

    base64_data = base64.b64encode(encoded[1])
    print(type(base64_data))

    base64_str = "data:image/jpg;base64,"+str(base64_data, 'utf-8')

    #print(base64_str)
    return base64_str



if __name__ == '__main__':
    path = 'test_imgs/test10.jpg'
    save_path = path + '_result.jpg'
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    model_path = 'model.ckpt'

    borders = findBorderContours(img)
    imgData = transMNIST(img, borders)
    results = predict(model_path, imgData)
    showResults(img, save_path, borders, results)
