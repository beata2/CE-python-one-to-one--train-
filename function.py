import tensorflow as tf
import numpy as np
import pandas as pd
import copy

preamble = 26
CSI_len = 16   #CSI长度
L_len = 512   #上行数据长度
Ek = 10**(0.5)       #发送能量
rou = 0.2     #叠加因子
walsh = pd.read_csv('walsh_16_512.csv')  # 1024x32
walsh = walsh.astype(np.float32)

arfa = 1.
batch = 200   #批次数
std = 0.01
lr = 0.00001    #学习速率0.00001
lr_2 = 0.000001   #学习速率0.000001
theta = 0.00002 #正则化系数
m = 10000
leng = 15
epoch = 20000
iteration = int(epoch*leng*m/batch + 1)

H_Net_1 = np.array([2*CSI_len, 16*CSI_len, 2*CSI_len])
D_Net_1 = np.array([2*L_len,16*L_len,2*L_len])
Const_h = np.float32((rou/CSI_len)**0.5)
Const_d = np.float32(((1-rou))**0.5)
Const_dd = np.float32(Const_d**(-1))

def choose_layer(layer,xs):
    if layer == 1:
        h_1 = despreading(xs)
        [out, w_1, b_1, w_2, b_2] = (H_DNN_1(h_1))
    elif layer == 2:
        h_1 = despreading(xs)
        [H_1, w_1, b_1, w_2, b_2] = (H_DNN_1_ok(h_1))
        d_1 = (tf.subtract(xs, tf.multiply(Const_h, spreading(H_1))))
        [out, w_1, b_1, w_2, b_2] = (D_DNN_1(d_1))
    elif layer == 3:
        h_1 = despreading(xs)
        [H_1, w_1, b_1, w_2, b_2] = (H_DNN_1_ok(h_1))
        d_1 = (tf.subtract(xs, tf.multiply(Const_h, spreading(H_1))))
        [D_1, w_1, b_1, w_2, b_2] = (D_DNN_1_ok(d_1))
        h_2 = despreading(tf.subtract(xs, tf.multiply(Const_d, D_1)))
        [out, w_1, b_1, w_2, b_2] = (H_DNN_2(h_2))
    elif layer == 4:
        h_1 = despreading(xs)
        [H_1, w_1, b_1, w_2, b_2] = (H_DNN_1_ok(h_1))
        d_1 = (tf.subtract(xs, tf.multiply(Const_h, spreading(H_1))))
        [D_1, w_1, b_1, w_2, b_2] = (D_DNN_1_ok(d_1))
        h_2 = despreading(tf.subtract(xs, tf.multiply(Const_d, D_1)))
        [H_2, w_1, b_1, w_2, b_2] = (H_DNN_2_ok(h_2))
        d_2 = (tf.subtract(xs, tf.multiply(Const_h, spreading(H_2))))
        [out, w_1, b_1, w_2, b_2] = (D_DNN_2(d_2))
    else:
        # 做为测试模型使用
        h_1 = despreading(xs)
        [H_1, a, b, c, d] = (H_DNN_1_ok(h_1))
        d_1 = (tf.subtract(xs, tf.multiply(Const_h, spreading(H_1))))
        [D_1, a, b, c, d] = (D_DNN_1_ok(d_1))
        h_2 = despreading(tf.subtract(xs, tf.multiply(Const_d, D_1)))
        [w_1, a, b, c, d] = (H_DNN_2_ok(h_2))        # **********为方便起见，这里的H_2用w_1代替*********
        d_2 = (tf.subtract(xs, tf.multiply(Const_h, spreading(w_1))))
        [out, a, b_1, w_2, b_2] = (D_DNN_2_ok(d_2))
    return [out, w_1, b_1, w_2, b_2]

def kuo_pin(x):
    Q = walsh        #扩频向量
    out = np.dot(x,np.transpose(Q))
    return out

def batch_norm(x):
    y = copy.copy(x)
    mean = tf.reduce_mean(y)
    y = (y - mean) / tf.sqrt(tf.reduce_mean(tf.square(y - mean)))
    return y

def despreading(x):  #x向量为实数向量
    y = tf.reshape(tf.stack([tf.matmul(x[:,:L_len],walsh),tf.matmul(x[:,L_len:],walsh)],axis=1),[-1,2*CSI_len])
    return y

def spreading(x):  #x向量为实数向量
    Real = tf.matmul(x[:,:CSI_len],np.transpose(walsh))
    Imag = tf.matmul(x[:,CSI_len:],np.transpose(walsh))
    return tf.reshape(tf.stack([Real,Imag],axis=1),[-1,2*L_len])

def data_yingshe(x):
    temp = copy.copy(x)
    shape = np.shape(temp)
    temp = np.reshape(temp, [1, -1])
    for ii in range(np.size(temp)):
        if (temp[0, ii] <= 0.0):
            temp[0, ii] = 0.0
        else:
            temp[0, ii] = 1.0
    return np.reshape(temp, shape)

def BER(x,y):
    num_x = np.size(x)
    temp = x-y
    num_temp = sum(sum(temp**2))
    return  num_temp/num_x

def sig_gen(M,N):
    data = (np.random.randint(0,2,[M,N])*2. - 1.) + 1j*(np.random.randint(0,2,[M,N])*2. - 1.)
    out = np.sqrt(1/2)*data
    return out

def H_DNN_1(input):
    w_1 = tf.Variable(tf.random_normal([H_Net_1[0],H_Net_1[1]],mean=0.0,stddev=std))
    b_1 = tf.Variable(tf.zeros([1,H_Net_1[1]]))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_1)
    tf.add_to_collection(tf.GraphKeys.BIASES, b_1)
    layer_1 = tf.nn.tanh((tf.add(tf.matmul(batch_norm(input), w_1), b_1)))
    w_2 = tf.Variable(tf.random_normal([H_Net_1[1],H_Net_1[2]],mean=0.0,stddev=std))
    b_2 = tf.Variable(tf.zeros([1,H_Net_1[2]]))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_2)
    tf.add_to_collection(tf.GraphKeys.BIASES, b_2)
    layer_2 = (tf.add(tf.matmul((layer_1),w_2),b_2))
    return [layer_2,w_1,b_1,w_2,b_2]

def H_DNN_1_ok(input):
    w_1 = np.float32(pd.read_csv('wh_11.csv').ix[:,1:])
    b_1 = np.float32(pd.read_csv('bh_11.csv').ix[:,1:])
    layer_1 = tf.nn.tanh((tf.add(tf.matmul(batch_norm(input), w_1), b_1)))
    w_2 = np.float32(pd.read_csv('wh_12.csv').ix[:,1:])
    b_2 = np.float32(pd.read_csv('bh_12.csv').ix[:,1:])
    layer_2 = (tf.add(tf.matmul((layer_1),w_2),b_2))
    return [layer_2,w_1,b_1,w_2,b_2]

def D_DNN_1(input):
    w_1 = tf.Variable(tf.random_normal([D_Net_1[0],D_Net_1[1]],mean=0.0,stddev=std))
    b_1 = tf.Variable(tf.zeros([1,D_Net_1[1]]))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_1)
    tf.add_to_collection(tf.GraphKeys.BIASES, b_1)
    layer_1 = tf.nn.tanh((tf.add(tf.matmul(batch_norm(input), w_1), b_1)))
    w_2 = tf.Variable(tf.random_normal([D_Net_1[1],D_Net_1[2]],mean=0.0,stddev=std))
    b_2 = tf.Variable(tf.zeros([1,D_Net_1[2]]))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_2)
    tf.add_to_collection(tf.GraphKeys.BIASES, b_2)
    layer_2 = (tf.add(tf.matmul((layer_1),w_2),b_2))
    return [layer_2,w_1,b_1,w_2,b_2]
def D_DNN_1_ok(input):
    w_1 = np.float32(pd.read_csv('wd_11.csv').ix[:,1:])
    b_1 = np.float32(pd.read_csv('bd_11.csv').ix[:,1:])
    layer_1 = tf.nn.tanh((tf.add(tf.matmul(batch_norm(input), (arfa * w_1)), (arfa * b_1))))
    w_2 = np.float32(pd.read_csv('wd_12.csv').ix[:,1:])
    b_2 = np.float32(pd.read_csv('bd_12.csv').ix[:,1:])
    layer_2 = 0.5**0.5*tf.nn.tanh(10000*(tf.add(tf.matmul((layer_1),w_2),b_2)))
    return [layer_2,w_1,b_1,w_2,b_2]


def H_DNN_2(input):
    w_1 = tf.Variable(tf.random_normal([H_Net_1[0],H_Net_1[1]],mean=0.0,stddev=std))
    b_1 = tf.Variable(tf.zeros([1,H_Net_1[1]]))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_1)
    tf.add_to_collection(tf.GraphKeys.BIASES, b_1)
    layer_1 = tf.nn.tanh((tf.add(tf.matmul(batch_norm(input), w_1), b_1)))
    w_2 = tf.Variable(tf.random_normal([H_Net_1[1],H_Net_1[2]],mean=0.0,stddev=std))
    b_2 = tf.Variable(tf.zeros([1,H_Net_1[2]]))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_2)
    tf.add_to_collection(tf.GraphKeys.BIASES, b_2)
    layer_2 = (tf.add(tf.matmul((layer_1),w_2),b_2))
    return [layer_2,w_1,b_1,w_2,b_2]

def H_DNN_2_ok(input):
    w_1 = np.float32(pd.read_csv('wh_21.csv').ix[:,1:])
    b_1 = np.float32(pd.read_csv('bh_21.csv').ix[:,1:])
    layer_1 = tf.nn.tanh((tf.add(tf.matmul(batch_norm(input), w_1), b_1)))
    w_2 = np.float32(pd.read_csv('wh_22.csv').ix[:,1:])
    b_2 = np.float32(pd.read_csv('bh_22.csv').ix[:,1:])
    layer_2 = (tf.add(tf.matmul((layer_1),w_2),b_2))
    return [layer_2,w_1,b_1,w_2,b_2]
#
def D_DNN_2(input):
    w_1 = tf.Variable(tf.random_normal([D_Net_1[0],D_Net_1[1]],mean=0.0,stddev=std))
    b_1 = tf.Variable(tf.zeros([1,D_Net_1[1]]))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_1)
    tf.add_to_collection(tf.GraphKeys.BIASES, b_1)
    layer_1 = tf.nn.tanh((tf.add(tf.matmul(batch_norm(input), w_1), b_1)))
    w_2 = tf.Variable(tf.random_normal([D_Net_1[1],D_Net_1[2]],mean=0.0,stddev=std))
    b_2 = tf.Variable(tf.zeros([1,D_Net_1[2]]))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_2)
    tf.add_to_collection(tf.GraphKeys.BIASES, b_2)
    layer_2 = (tf.add(tf.matmul((layer_1),w_2),b_2))
    return [layer_2,w_1,b_1,w_2,b_2]

def D_DNN_2_ok(input):
    w_1 = np.float32(pd.read_csv('wd_21.csv').ix[:,1:])
    b_1 = np.float32(pd.read_csv('bd_21.csv').ix[:,1:])
    layer_1 = tf.nn.tanh((tf.add(tf.matmul(batch_norm(input), (arfa * w_1)), (arfa * b_1))))
    w_2 = np.float32(pd.read_csv('wd_22.csv').ix[:,1:])
    b_2 = np.float32(pd.read_csv('bd_22.csv').ix[:,1:])
    layer_2 = 0.5**0.5*tf.nn.tanh(10000*tf.add(tf.matmul((layer_1),w_2),b_2))
    return [layer_2,w_1,b_1,w_2,b_2]

def data_ce(m,Ek):
    # 信道估计出错
    T_send_temp = []
    T_CSI_temp = []
    T_data_temp = []
    for ii in range(m):
        L_1 = sig_gen(1, L_len)
        CSI = 0.5 ** 0.5 * (np.random.normal(0, 1, [1, CSI_len]) + 1j * np.random.normal(0, 1, [1, CSI_len]))
        CSI_kuo = kuo_pin(CSI)
        T_send = (np.sqrt(1 - rou) * L_1 + np.sqrt(rou / CSI_len) * CSI_kuo)
        G = ((CSI_len) ** (-0.5)) * (0.5 ** 0.5) * (np.random.normal(0, 1, [CSI_len, 1]) + 1j * np.random.normal(0, 1, [CSI_len, 1]))

        # 用LS估计
        polit = np.sqrt(1 / 2) * (np.random.randint(0, 2, [1, preamble]) * 2. - 1.) + 1j * (np.random.randint(0, 2, [1, preamble]) * 2. - 1.)
        y_polit = np.dot(G,polit)+(Ek ** -0.5) * (0.5 ** 0.5) * (np.random.normal(0, 1, [CSI_len, preamble]) + 1j * np.random.normal(0, 1, [CSI_len, preamble]))
        G_LS = np.dot(y_polit,np.linalg.pinv(polit))

        N_mat = (Ek ** -0.5) * (0.5 ** 0.5) * (np.random.normal(0, 1, [CSI_len, L_len]) + 1j * np.random.normal(0, 1, [CSI_len, L_len]))
        N_temp = np.dot(np.linalg.pinv(G_LS), N_mat)
        GGX_temp = np.dot(np.dot(np.linalg.pinv(G_LS), G), T_send)
        T_send_mat = GGX_temp + N_temp
        T_send_temp.append(T_send_mat)
        T_CSI_temp.append(CSI)
        T_data_temp.append(L_1)
    T_data_1 = np.reshape(T_send_temp, [m, -1])
    CSI_1 = np.reshape(T_CSI_temp, [m, -1])
    data_1 = np.reshape(T_data_temp, [m, -1])
    return T_data_1,CSI_1,data_1

def data_chacuolv(m,Ek):
    # 人为设定信道差错率
    T_send_temp = []
    T_CSI_temp = []
    T_data_temp = []
    for ii in range(m):
        L_1 = sig_gen(1, L_len)
        CSI = 0.5 ** 0.5 * (np.random.normal(0, 1, [1, CSI_len]) + 1j * np.random.normal(0, 1, [1, CSI_len]))
        CSI_kuo = kuo_pin(CSI)
        T_send = (np.sqrt(1 - rou) * L_1 + np.sqrt(rou / CSI_len) * CSI_kuo)
        G = ((CSI_len) ** (-0.5)) * (0.5 ** 0.5) * (np.random.normal(0, 1, [CSI_len, 1]) + 1j * np.random.normal(0, 1, [CSI_len, 1]))
        derta_G = 1 / 1000 * ((CSI_len) ** (-0.5)) * (0.5 ** 0.5) * ( np.random.normal(0, 1, [CSI_len, 1]) + 1j * np.random.normal(0, 1, [CSI_len, 1]))
        # derta_G = 0  # 当derta为0时，认为信道不出错
        G_CE = G + derta_G
        HHX_temp = np.dot(np.dot(np.linalg.pinv(G_CE), G), T_send)

        N_mat = (Ek ** -0.5) * (0.5 ** 0.5) * (np.random.normal(0, 1, [CSI_len, L_len]) + 1j * np.random.normal(0, 1, [CSI_len, L_len]))
        N_temp = np.dot(np.linalg.pinv(G_CE), N_mat)

        T_send_mat = HHX_temp + N_temp
        T_send_temp.append(T_send_mat)
        T_CSI_temp.append(CSI)
        T_data_temp.append(L_1)
    T_data_1 = np.reshape(T_send_temp, [m, -1])
    CSI_1 = np.reshape(T_CSI_temp, [m, -1])
    data_1 = np.reshape(T_data_temp, [m, -1])
    return T_data_1,CSI_1,data_1

#*********************************生成训练数据集***************************************
def gen_data(fun):
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()
    for ii in range(leng):
        [T_data_1,CSI_1,data_1] = fun(m,Ek)
        T_data = np.hstack((np.real(T_data_1), np.imag(T_data_1)))  # 神经网络输入   按行拼接  [m,2L]
        CSI_data = (np.hstack((np.real(CSI_1), np.imag(CSI_1))))  # CSI  [m,2CSI]
        L_data = (np.hstack((np.real(data_1), np.imag(data_1))))  # [m,2L]
        input_CSI_data = T_data
        output_CSI = CSI_data  # 神经网络输出
        output_data = L_data

        # 导出数据为.CSV文件格式
        x_val = pd.DataFrame(input_CSI_data)
        y_CSI = pd.DataFrame(output_CSI)
        y_data = pd.DataFrame(output_data)

        df1 = df1.append(x_val)
        df2 = df2.append(y_CSI)
        df3 = df3.append(y_data)
        print(ii)
    print(np.shape(df1))
    print(np.shape(df2))
    print(np.shape(df3))
    print(output_CSI)
    print('-' * 50)
    print(output_data)
    print('-' * 50)
    print("导出x_val")
    df1.to_csv('x_val.csv')
    print("导出y_CSI")
    df2.to_csv('y_CSI.csv')
    print("导出y_data")
    df3.to_csv('y_data.csv')

