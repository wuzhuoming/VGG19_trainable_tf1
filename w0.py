import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import vgg19_trainable as vgg19
import utils

tf.app.flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224", "worker hosts")
tf.app.flags.DEFINE_string("job_name", "worker", "'ps' or'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("num_workers", 2, "Number of workers")
tf.app.flags.DEFINE_boolean("is_sync", True, "using synchronous training or not")

FLAGS = tf.app.flags.FLAGS



def main(_):
    worker_hosts = FLAGS.worker_hosts.split(",")

    # create the cluster configured by `ps_hosts' and 'worker_hosts'
    cluster = tf.train.ClusterSpec({"worker": worker_hosts})

    # create a server for local task
    server = tf.train.Server(cluster, job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % (FLAGS.task_index),
                                                      cluster=cluster)):
            img1 = utils.load_image("./test_data/tiger.jpeg")
            img1_true_result = [1 if i == 292 else 0 for i in range(1000)]  # 1-hot result for tiger
            batch1 = img1.reshape((1, 224, 224, 3))

            images = tf.placeholder(tf.float32, [1, 224, 224, 3])
            true_out = tf.placeholder(tf.float32, [1, 1000])
            train_mode = tf.placeholder(tf.bool)

            vgg = vgg19.Vgg19('./vgg19.npy')
            vgg.build(images, train_mode)

            print(vgg.get_var_count())

            # The StopAtStepHook handles stopping after running given steps.
            hooks = [tf.train.StopAtStepHook(last_step=10000)]

            global_step = tf.train.get_or_create_global_step()
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-04)

            if FLAGS.is_sync:
                # asynchronous training
                # use tf.train.SyncReplicasOptimizer wrap optimizer
                # ref: https://www.tensorflow.org/api_docs/python/tf/train/SyncReplicasOptimizer
                optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=FLAGS.num_workers,
                                                           total_num_replicas=FLAGS.num_workers)
                # create the hook which handles initialization and queues
                hooks.append(optimizer.make_session_run_hook((FLAGS.task_index == 0)))

            loss = tf.reduce_sum((vgg.prob - true_out) ** 2)
            train_op = optimizer.minimize(loss, global_step=global_step,
                                          aggregation_method=tf.AggregationMethod.ADD_N)

            # The MonitoredTrainingSession takes care of session initialization,
            # restoring from a checkpoint, saving to a checkpoint, and closing when done
            # or an error occurs.
            with tf.train.MonitoredTrainingSession(master=server.target,
                                                   is_chief=(FLAGS.task_index == 0),
                                                   checkpoint_dir="./checkpoint_dir",
                                                   hooks=hooks) as mon_sess:
                # mon_sess.run(tf.global_variables_initializer())
                while not mon_sess.should_stop():
                    _, prob, step = mon_sess.run([train_op, vgg.prob, global_step],
                                               feed_dict={images: batch1, true_out: [img1_true_result],
                                                           train_mode: True})
                    if step % 100 == 0:
                        print("Train step %d" % step)
                        utils.print_prob(prob[0], './synset.txt')


if __name__ == "__main__":
    tf.app.run()
