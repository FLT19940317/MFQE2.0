import numpy as np
import tensorflow as tf
import os, time
import net_MFCNN, flow


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # only show error and warning
config = tf.ConfigProto(allow_soft_placement = True) # if GPU is not usable, then turn to CPU automatically

BATCH_SIZE = 1
CHANNEL = 1

def enhance(QP,input,Non_PQF_indices,pre_PQF_indices,sub_PQF_indices):

    # Recommended MF-CNN model
    if QP == 22:
        model_index = 1230000
    elif QP == 27:
        model_index = 275000
    elif QP == 32:
        model_index = 1227500
    elif QP == 37:
        model_index = 1122500
    elif QP == 42:
        model_index = 1250000

    model_path = "./Model_MFCNN/QP" + str(QP) + "/model.ckpt-" + str(model_index)

    # video information
    nfs = len(Non_PQF_indices)
    height = input.shape[1]
    width = input.shape[2]

    input = input[:,:,:,np.newaxis]
    input = input / 255.0

    enhanced_frames = np.zeros([nfs,height,width])

    with tf.Graph().as_default() as g:

        x1 = tf.placeholder(tf.float32, [BATCH_SIZE, height, width, CHANNEL])  # previous
        x2 = tf.placeholder(tf.float32, [BATCH_SIZE, height, width, CHANNEL])  # current
        x3 = tf.placeholder(tf.float32, [BATCH_SIZE, height, width, CHANNEL])  # subsequent
        is_training = tf.placeholder_with_default(False, shape=())

        x1to2 = flow.warp_img(BATCH_SIZE, x2, x1, False)
        x3to2 = flow.warp_img(BATCH_SIZE, x2, x3, True)

        if (QP == 37) or (QP == 42):
            x2_enhanced = net_MFCNN.network(x1to2, x2, x3to2, is_training)
        elif (QP == 22) or (QP == 27) or (QP == 32):
            x2_enhanced = net_MFCNN.network2(x1to2, x2, x3to2)

        # restore vars above
        saver = tf.train.Saver()

        with tf.Session(config = config) as sess:

            # restore model
            saver.restore(sess,model_path)

            # enhance
            start_time = time.time()
            for ite in range(nfs):
                x1_feed = input[pre_PQF_indices[ite]:pre_PQF_indices[ite]+1,:,:,:]
                x2_feed = input[Non_PQF_indices[ite]:Non_PQF_indices[ite]+1,:,:,:]
                x3_feed = input[sub_PQF_indices[ite]:sub_PQF_indices[ite]+1,:,:,:]

                x2_enhanced_frame = sess.run(x2_enhanced, feed_dict={x1: x1_feed, x2: x2_feed, x3: x3_feed, is_training:False})
                enhanced_frames[ite] = np.squeeze(x2_enhanced_frame)

                print("\r"+str(ite+1)+" | "+str(nfs), end="", flush=True)

            end_time = time.time()
            average_fps = nfs / (end_time - start_time)
            print("")

    return enhanced_frames, average_fps
