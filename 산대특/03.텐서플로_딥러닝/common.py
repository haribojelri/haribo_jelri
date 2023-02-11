import numpy as np
from tensorflow.keras.datasets import mnist
# 함수로 작성
# 정규화 옵션, one hot encording, flatten 
def loadMnist(nomalize = False, flatten = False, onehotEncording=False):
    # 이미지 정규화  0 ~ 255  
    (x_train,y_train),(x_test,y_test) =  mnist.load_data()
    if nomalize:
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0
    if not  flatten:
        x_train = x_train.reshape(-1,1,28,28)
        x_test  = x_test.reshape(-1,1,28,28)
        
    if onehotEncording: 
        result_train_y,result_test_y = [],[]
        for i in y_train:
            t = np.zeros(10)
            t[i] = 1
            result_train_y.append(t)
        for i in y_test:
            t = np.zeros(10)
            t[i] = 1
            result_test_y.append(t)
        y_train = np.array(result_train_y)
        y_test = np.array(result_test_y)
            
    
    return (x_train,y_train),(x_test,y_test)
    

def sigmoid(x):
    x = np.array(x)
    return 1 / (1 + np.exp(-x))

def relu(x):
    x = np.array(x)
    return np.maximum(0,x)

class Relu:
    def __init__(self):
        self.mask = None
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx
    
def softmax(x):
    x = np.array(x)
    x = x-np.max(x) # 오버플로우 방지
    y = np.exp(x) / np.sum(np.exp(x))
    return y

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None #손실
        self.y = None # softmax 출력
        self.t = None # 정답 레이블(원 핫 벡터)
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx

def cross_entropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def f(w):
    return crossentropy(softmax(x@w),t)

def numeric_gradient(f,x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 값 복원
        it.iternext()   
        
    return grad
    
def im2col(input_data, filter_h, filter_w, stride=1,pad=0):
    '''
    input_data : 4차원(이미지수, 채널수, 높이, 너비)
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    
    returns : 2차원 배열    
    '''
    N,C,H,W =  input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    
    img = np.pad(input_data, [(0,0),(0,0),(pad,pad),(pad,pad)], 'constant',constant_values=0)
    col = np.zeros( (N,C, filter_h,filter_w,out_h,out_w) )
    
    for y in range(filter_h):
        y_max = y+stride*out_h
        for x in range(filter_w):
            x_max = x+stride*out_w
            col[:,:,y,x, :, :] = img[:,:,y:y_max:stride, x:x_max:stride]
    col = col.transpose(0,4,5,1,2,3).reshape(N*out_h*out_w,-1)
    return col       

def col2im(col, input_shape,filter_h, filter_w, stride=1,pad=0):
    '''
    2차원 배열을 받아서 이미지의 묶음으로 변환
    col : 2차원 배열
    input_shape : ex  (10,1,28,28)    
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    
    returns : 변경된 이미지들..
    '''
    N,C,H,W =  input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    
    col = col.reshape(N, out_h, out_w, C, filter_h,filter_w).transpose(0,3,4,5,1,2)
    img = np.zeros( (N,C,H+2*pad+stride-1, W+2*pad+stride-1 ) )
    
    
    
    for y in range(filter_h):
        y_max = y+stride*out_h
        for x in range(filter_w):
            x_max = x+stride*out_w
            img[:,:,y:y_max:stride, x:x_max:stride] += col[:, :, y, x, : , :]                
    return img[:,:, pad: H + pad, pad : W + pad ]


class Affine:
    def __init__(self,w,b):
        self.w = w
        self.b = b
        self.x = None
        self.dw = None
        self.db = None
    def forward(self, x):
        self.original_x_shape = x.shape
        x  = x.reshape(x.shape[0], -1)
        self.x = x
        out = x@self.w + self.b
        return out
    def backward(self, dout):
        dx = np.dot(dout, self.w.T) 
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout,axis=0)
        dx = dx.reshape(*self.original_x_shape)
        return dx