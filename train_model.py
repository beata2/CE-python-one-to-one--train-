from function import *
sess = tf.Session()
xs = tf.placeholder(tf.float32, shape=[None, 2 * L_len])
target_H = tf.placeholder(tf.float32, shape=[None, 2 * CSI_len])
target_D = tf.placeholder(tf.float32, shape=[None, 2 * L_len])


def train_layer(layer):
    [out, w_1, b_1, w_2, b_2] = choose_layer(layer,xs)
    #*************************训练*********************************************
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

#******************************再训练***************************************
def train_layer_again(layer):
    file_name = "./t/Unit2-model-2000"
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
# train_layer(1)
train_layer_again(1)