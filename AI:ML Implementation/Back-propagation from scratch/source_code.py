# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

#生成資料
def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2)) #從[0, 1]取出
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    inputs = []
    labels = []
    
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        
        if 0.1*i == 0.5:
            continue
        
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    
    return np.array(inputs), np.array(labels).reshape(21, 1)

#顯示結果
def show_result(x, y, pred_y):
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if round(float(pred_y[i])) == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()

#AC rate
def accuracy(x, y, n):
    total = 0
    for i in range(len(x)):
        bias = x[i] - y[i]
        total += abs(bias)
    return 1 - float(total / n)

#2 layer Neural Network
class Neural_Network(object):
    def __init__(self):
        #各layer大小設定
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3

        #初始權重
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) 
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) 

    def forward(self, x):

        self.z1 = np.dot(x, self.W1) 
        self.z2 = self.sigmoid(self.z1) 
        self.z = np.dot(self.z2, self.W2) 
        output = self.sigmoid(self.z) 
        return output 

    def sigmoid(self, x):
        # 激勵函數 
        return 1/(1+np.exp(-x))

    def sigmoidPrime(self, x):
        # 激勵函數微分
        return x * (1 - x)

    def backward(self, x, y, output):

        self.error_output = y - output 
        self.delta_output = self.error_output*self.sigmoidPrime(output) 

        self.z2_error = self.delta_output.dot(self.W2.T) 
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) 

        self.W1 += x.T.dot(self.z2_delta) 
        self.W2 += self.z2.T.dot(self.delta_output) 

    def train (self, x, y):
        output = self.forward(x)
        self.backward(x, y, output)

def main():

    NN1 = Neural_Network()
    NN2 = Neural_Network()

    x1, y1 = generate_linear(n=100)
    x2, y2 = generate_XOR_easy()

    # linear train NN1 
    for i in range(10001): # trains the NN 20,001 times
        if i % 1000 == 0:
            print ("linear train - epoch "+str(i)+" loss : " + str(np.mean(np.square(y1 - NN1.forward(x1))))) # mean sum squared loss
        NN1.train(x1, y1)

    # 先用XOR train NN2 
    for i in range(10001): # trains the NN 20,001 times
        if i % 1000 == 0:
            print ("XOR train - epoch "+str(i)+" loss : " + str(np.mean(np.square(y2 - NN2.forward(x2))))) # mean sum squared loss
        NN2.train(x2, y2)

    #把線性資料的訓練結果show出來
    show_result(x1,y1,NN1.forward(x1))

    #把XOR資料的訓練結果show出來
    show_result(x2,y2,NN2.forward(x2))

    print ("The accuracy rate of linear dataset is: "+ str(accuracy(NN1.forward(x1), y1, 100)))
    print ("The accuracy rate of XOR dataset is: "+ str(accuracy(NN2.forward(x2), y2, 100)))

 
if __name__ == '__main__':
    main()
