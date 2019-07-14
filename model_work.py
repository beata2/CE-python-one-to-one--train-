from function import *
file_name = "./t/Unit2-model-5000"
sess = tf.Session()
xs = tf.placeholder(tf.float32, shape=[None, 2 * L_len])
target_H = tf.placeholder(tf.float32, shape=[None, 2 * CSI_len])
target_D = tf.placeholder(tf.float32, shape=[None, 2 * L_len])

# *******************************************************训练************************************************************
def train_layer(layer):
    [out, w_1, b_1, w_2, b_2] = choose_layer(layer,xs)
    regularizer = tf.contrib.layers.l2_regularizer(theta)  # 正则化参数
    reg_term = tf.contrib.layers.apply_regularization(regularizer)
    if layer == 1 or layer == 3:
        loss_H = tf.reduce_mean(tf.square(target_H - out))/tf.reduce_mean(tf.square(target_H))
        loss = loss_H + reg_term
    else:
        loss_D = tf.reduce_mean(tf.square(target_D - out)) / tf.reduce_mean(tf.square(target_D))
        loss = loss_D + reg_term
    train = tf.train.AdamOptimizer(lr).minimize(loss)

    x_ = pd.read_csv('x_val.csv')
    H_ = pd.read_csv('y_CSI.csv')
    D_ = pd.read_csv('y_data.csv')

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    saver = tf.train.Saver(max_to_keep=4)
    for ii in range(iteration):
        rand_index = np.random.choice(len(x_), size=batch)
        x_input = x_.ix[rand_index, 1:]
        H_output = H_.ix[rand_index, 1:]
        D_output = D_.ix[rand_index, 1:]
        feed_dict = {xs: x_input, target_H: H_output, target_D: D_output}
        sess.run(train, feed_dict=feed_dict)
        if ii % 1000 == 0:
            if layer == 1 or layer == 3:
                print('训练%d后，loss_H = %8f.' % (ii, sess.run(loss_H, feed_dict=feed_dict)))
            else:
                out_data = data_yingshe(sess.run(out, feed_dict=feed_dict))
                ber = BER(out_data, data_yingshe(sess.run(target_D, feed_dict=feed_dict)))
                print('-' * 50)
                print('训练%d后，loss_D = %8f.' % (ii, sess.run(loss_D, feed_dict=feed_dict)))
                print('训练%d后，BER = %12f.' % (ii,ber))

            # print('训练%d后，总误差Loss = %8f.' % (ii, sess.run(loss, feed_dict=feed_dict)))
            saver.save(sess, 't/Unit2-model', global_step=ii)

#******************************************************再训练***************************************************************
def train_layer_again(layer):
    [out, w_1, b_1, w_2, b_2] = choose_layer(layer, xs)
    regularizer = tf.contrib.layers.l2_regularizer(theta)  # 正则化参数
    reg_term = tf.contrib.layers.apply_regularization(regularizer)

    if layer == 1 or layer == 3:
        loss_H = tf.reduce_mean(tf.square(target_H - out))/tf.reduce_mean(tf.square(target_H))
        loss = loss_H + reg_term
    else:
        loss_D = tf.reduce_mean(tf.square(target_D - out)) / tf.reduce_mean(tf.square(target_D))
        loss = loss_D + reg_term
    train = tf.train.AdamOptimizer(lr).minimize(loss)

    x_ = pd.read_csv('x_val.csv')
    H_ = pd.read_csv('y_CSI.csv')
    D_ = pd.read_csv('y_data.csv')

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    saver = tf.train.Saver()
    saver.restore(sess,file_name)
    saver = tf.train.Saver(max_to_keep=4)
    for ii in range(iteration):
        rand_index = np.random.choice(len(x_), size=batch)
        x_input = x_.ix[rand_index, 1:]
        H_output = H_.ix[rand_index, 1:]
        D_output = D_.ix[rand_index, 1:]
        feed_dict = {xs: x_input, target_H: H_output, target_D: D_output}
        sess.run(train, feed_dict=feed_dict)
        if ii % 1000 == 0:
            if layer == 1 or layer == 3:
                print('再训练%d后，loss_H = %8f.' % (ii, sess.run(loss_H, feed_dict=feed_dict)))
            else:
                out_data = data_yingshe(sess.run(out, feed_dict=feed_dict))
                ber = BER(out_data, data_yingshe(sess.run(target_D, feed_dict=feed_dict)))
                print('-' * 50)
                print('再训练%d后，loss_D = %8f.' % (ii, sess.run(loss_D, feed_dict=feed_dict)))
                print('再训练%d后，BER = %12f.' % (ii, ber))

            # print('训练%d后，总误差Loss = %8f.' % (ii, sess.run(loss, feed_dict=feed_dict)))
            saver.save(sess, 't/Unit2-model', global_step=ii)
#******************************************************保存***************************************************************
def save_layer(layer):
    [out, w_1, b_1, w_2, b_2] = choose_layer(layer,xs)
    saver = tf.train.Saver()
    saver.restore(sess, file_name)

    df_w1 = pd.DataFrame(sess.run(w_1))
    df_b1 = pd.DataFrame(sess.run(b_1))
    df_w2 = pd.DataFrame(sess.run(w_2))
    df_b2 = pd.DataFrame(sess.run(b_2))
    if layer == 1:
        df_w1.to_csv('wh_11.csv')
        df_b1.to_csv('bh_11.csv')
        df_w2.to_csv('wh_12.csv')
        df_b2.to_csv('bh_12.csv')
        print("save_H_1,OK")
    elif layer == 2:
        df_w1.to_csv('wd_11.csv')
        df_b1.to_csv('bd_11.csv')
        df_w2.to_csv('wd_12.csv')
        df_b2.to_csv('bd_12.csv')
        print("save_D_1,OK")
    elif layer == 3:
        df_w1.to_csv('wh_21.csv')
        df_b1.to_csv('bh_21.csv')
        df_w2.to_csv('wh_22.csv')
        df_b2.to_csv('bh_22.csv')
        print("save_H_2,OK")
    else:
        df_w1.to_csv('wd_21.csv')
        df_b1.to_csv('bd_21.csv')
        df_w2.to_csv('wd_22.csv')
        df_b2.to_csv('bd_22.csv')
        print("save_D_2,OK")
# ******************************************************测试***************************************************************
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

    feed_dict = {xs: input_CSI_data, target_H: output_CSI, target_D: output_data}
    out_data = data_yingshe(sess.run(D_2, feed_dict=feed_dict))
    ber = BER(out_data, data_yingshe(sess.run(target_D, feed_dict=feed_dict)))
    print("MSE-CSI：")
    mse = sess.run(loss_H, feed_dict=feed_dict)
    print(mse)
    print("BER-data：")
    print(ber)
    return [ber,mse]