from function import *
sess = tf.Session()
xs = tf.placeholder(tf.float32, shape=[None, 2 * L_len])
target_H = tf.placeholder(tf.float32, shape=[None, 2 * CSI_len])
target_D = tf.placeholder(tf.float32, shape=[None, 2 * L_len])
#******************************测试**************************************************
def test_model(layer,fun,SNR):   #layer不能为1-4
    Ek = 10 ** (0.1 * SNR)
    [D_2, H_2, aa, bb, cc] = choose_layer(layer, xs)
    [T_data_1, CSI_1, data_1] = fun(m,Ek)
    T_data = np.hstack((np.real(T_data_1), np.imag(T_data_1)))  # 神经网络输入   按行拼接  [m,2L]
    CSI_data = (np.hstack((np.real(CSI_1), np.imag(CSI_1))))  # CSI  [m,2CSI]
    L_data = (np.hstack((np.real(data_1), np.imag(data_1))))  # [m,2L]
    input_CSI_data = T_data
    output_CSI = CSI_data  # 神经网络输出
    output_data = L_data

    loss_H = tf.reduce_mean(tf.square(target_H - H_2))/tf.reduce_mean(tf.square(target_H))
    loss_D = tf.reduce_mean(tf.square(target_D - D_2))/tf.reduce_mean(tf.square(target_D))

    with tf.Session() as sess:
        feed_dict = {xs: input_CSI_data, target_H: output_CSI, target_D: output_data}

        out_data = data_yingshe(sess.run(D_2, feed_dict=feed_dict))
        ber = BER(out_data, data_yingshe(sess.run(target_D, feed_dict=feed_dict)))
        print('-' * 50)
        print("MSE-CSI：")
        # mse =
        print(sess.run(loss_H, feed_dict=feed_dict))
        print("BER-data：")
        print(ber)
        print('*' * 50)
test_model(5,data_ce,6)
test_model(5,data_ce,8)
