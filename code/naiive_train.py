import sys, os, argparse, gzip
import numpy as np
import tensorflow as tf
from adv_model import classifier
from get_data import get_data
height=28;
width=28;
num_channel=3;
num_label=10;

def train():
  # get data
  tr_d,tr_l,ts_d,ts_l=get_data();
  tr_d=np.concatenate([tr_d,tr_d,tr_d],3);
  ts_d=np.concatenate([ts_d,ts_d,ts_d],3);

  # input data
  img=tf.placeholder(tf.float32,[None,height,width,num_channel],name="img");
  label=tf.placeholder(tf.float32,[None,num_label],name="label");
   
  # classifier
  c_params,c_net,c_prediction=classifier(img); 
  
  # loss function and accuracy
  loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=c_net,labels=label));
  acc=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(c_prediction,1),tf.argmax(label,1)),tf.float32));

  # optimizer
  optimizer=tf.train.AdamOptimizer(0.001);
  trainer = optimizer.minimize(loss);

  # shuffle the training set
  tr_idx=range(len(tr_d)); np.random.shuffle(tr_idx);
  tr_d=tr_d[tr_idx]; tr_l=tr_l[tr_idx];

  # saver
  saver = tf.train.Saver();

  init = tf.global_variables_initializer();
  with tf.Session() as sess:
    sess.run(init);
    # tf board
    tf.summary.scalar('loss',loss);
    merged = tf.summary.merge_all();
    train_writer = tf.summary.FileWriter(FLAGS.tb_folder, sess.graph);
    for i in range(FLAGS.max_epoch):
      tr_acc=0;
      for j in range(len(tr_d)/FLAGS.batch_num):
        batch_img=tr_d[j*FLAGS.batch_num:(j+1)*FLAGS.batch_num];
        batch_label=tr_l[j*FLAGS.batch_num:(j+1)*FLAGS.batch_num];
        summary,tr_acc_b,_=sess.run([merged,acc,trainer],feed_dict={img:batch_img,label:batch_label});
        tr_acc+=tr_acc_b*FLAGS.batch_num;
        #train_writer.add_summary(summary,i*len(tr_d)+j*FLAGS.batch_num);
      ts_acc=sess.run([acc],feed_dict={img:ts_d,label:ts_l})[0];
      print(str(i+1)+" Epoch Training Acc: "+str(tr_acc/len(tr_d))+", Test Acc: "+str(ts_acc));
      # shuffle
      tr_idx=range(len(tr_d)); np.random.shuffle(tr_idx);
      tr_d=tr_d[tr_idx]; tr_l=tr_l[tr_idx];
    saver.save(sess,os.path.join(FLAGS.save_folder,FLAGS.save_file_name));

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch_num', type=int, default=1000,
                      help='the batch size')
  parser.add_argument('--max_epoch', type=int, default=50,
                      help='the max iteration')
  parser.add_argument('--tb_folder', type=str, default="/tmp/adv_training_cons/just_train",
                      help='the tensorboard log folder')
  parser.add_argument('--save_folder', type=str, default="/tmp/adv_training_cons/models/",
                      help='the model save folder')
  parser.add_argument('--save_file_name', type=str, default="just_trained.ckpt",
                      help='the model save file name')
  FLAGS, unparsed = parser.parse_known_args()
  train()

