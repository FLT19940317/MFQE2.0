import tensorflow as tf
import numpy as np
import os, time
import net_DSCNN


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # only show error and warning
config = tf.ConfigProto(allow_soft_placement = True) # if GPU is not usable, then turn to CPU automatically

BATCH_SIZE = 1
CHANNEL = 1


def enhance(QP,input):

    # Recommended DS-CNN model
    if QP == 22:
        model_index = 44000
    elif QP == 27:
        model_index = 27000
    elif QP == 32:
        model_index = 36000
    elif QP == 37:
        model_index = 11000
    elif QP == 42:
        model_index = 8000

    model_path = "./Model_DSCNN/QP" + str(QP) + "/model.ckpt-" + str(model_index)

    # video information
    nfs = input.shape[0]
    height = input.shape[1]
    width = input.shape[2]

    input = input[:,:,:,np.newaxis]
    input = input / 255.0

    enhanced_frames = np.zeros([nfs,height,width])

    with tf.Graph().as_default() as g:

        cmp_test = tf.placeholder(tf.float32,[BATCH_SIZE, height, width, CHANNEL])
        enhanced_test = net_DSCNN.network(cmp_test)

        # restore vars above
        saver = tf.train.Saver()

        with tf.Session(config = config) as sess:

            # restore model
            saver.restore(sess,model_path)

            # enhance
            start_time = time.time()
            for ite in range(nfs):
                cmp_frame = input[ite:ite+1,:,:,:]
                enhanced_frame = sess.run(enhanced_test, feed_dict={cmp_test:cmp_frame})
                enhanced_frames[ite] = np.squeeze(enhanced_frame)

                print("\r"+str(ite+1)+" | "+str(nfs), end="", flush=True)

            end_time = time.time()
            average_fps = nfs / (end_time - start_time)
            print("")

    return enhanced_frames, average_fps
