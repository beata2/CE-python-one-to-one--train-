from function import *
file_name = "./t/Unit2-model-27000"
sess = tf.Session()
xs = tf.placeholder(tf.float32, shape=[None, 2 * L_len])
target_H = tf.placeholder(tf.float32, shape=[None, 2 * CSI_len])
target_D = tf.placeholder(tf.float32, shape=[None, 2 * L_len])


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
save_layer(4)

