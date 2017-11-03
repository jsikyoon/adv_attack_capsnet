import sys, os, argparse, gzip
import numpy as np
import tensorflow as tf
from adv_model import classifier
from get_data import get_data
from skimage.io import imsave
from cleverhans.attacks import FastGradientMethod

FLAGS=None;
height=28;
width=28;
num_channel=3;
num_label=10;

class classifier_model(object):
  """Model class for CleverHans library."""

  def __init__(self):
    self.num_classes=num_label;
    self.built=False;
    img=tf.placeholder(tf.float32,[FLAGS.batch_num*2,height,width,num_channel],name="img");
    c_params,c_net,c_prediction=classifier(img);
    self.net=c_net;
    self.img=img;
    self.params=c_params;
    self.prediction=c_prediction;

  def __call__(self,img):
    self.built=True;
    return self.prediction;

def train():
  # max epsilon
  eps = 2.0 * FLAGS.max_epsilon / 256.0 /FLAGS.max_iter

  # get data
  tr_d,tr_l,ts_d,ts_l=get_data();
  tr_d=np.concatenate([tr_d,tr_d,tr_d],3);
  ts_d=np.concatenate([ts_d,ts_d,ts_d],3);

  model=classifier_model();
  fgsm=FastGradientMethod(model);
  x_adv=fgsm.generate(model.img,eps=eps,clip_min=-1, clip_max=1.);
  c_prediction=model.prediction;
  c_net=model.net;
  img=model.img;

  c_prediction_org=tf.slice(c_prediction,[0,0],[FLAGS.batch_num,-1],name=None);
  c_prediction_adv=tf.slice(c_prediction,[FLAGS.batch_num,0],[FLAGS.batch_num,-1],name=None);
  c_net_org=tf.slice(c_net,[0,0],[FLAGS.batch_num,-1],name=None);
  c_net_adv=tf.slice(c_net,[FLAGS.batch_num,0],[FLAGS.batch_num,-1],name=None);

  # input data
  label=tf.placeholder(tf.float32,[FLAGS.batch_num*2,num_label],name="label");
  label_org=tf.slice(label,[0,0],[FLAGS.batch_num,-1],name=None); 
  label_adv=tf.slice(label,[FLAGS.batch_num,0],[FLAGS.batch_num,-1],name=None); 
  
  # loss function and accuracy
  loss_org=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=c_net_org,labels=label_org));
  acc_org=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(c_prediction_org,1),tf.argmax(label_org,1)),tf.float32));
  loss_adv=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=c_net_adv,labels=label_adv));
  acc_adv=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(c_prediction_adv,1),tf.argmax(label_adv,1)),tf.float32));

  # optimizer
  optimizer=tf.train.AdamOptimizer(0.001);
  trainer = optimizer.minimize(loss_org+loss_adv);

  # shuffle the training set
  tr_idx=range(len(tr_d)); np.random.shuffle(tr_idx);
  tr_d=tr_d[tr_idx]; tr_l=tr_l[tr_idx];

  # saver
  saver = tf.train.Saver(model.params);
  
  init = tf.global_variables_initializer();
  with tf.Session() as sess:
    sess.run(init);
    # tf board
    tf.summary.scalar('loss_org',loss_org);
    merged = tf.summary.merge_all();
    train_writer = tf.summary.FileWriter(FLAGS.tb_folder, sess.graph);
    #adv_img_set=[];adv_img_set_label=[];
    for i in range(FLAGS.max_epoch):
      """
      for j in range(len(adv_img_set)/(FLAGS.batch_num*2)):
        batch_img=adv_img_set[j*(FLAGS.batch_num*2):(j+1)*(FLAGS.batch_num*2)];
        batch_label=adv_img_set_label[j*(FLAGS.batch_num*2):(j+1)*(FLAGS.batch_num*2)];
        sess.run(trainer,feed_dict={img:batch_img,label:batch_label});
      """
      tr_acc_org=0;tr_acc_adv=0;
      for j in range(len(tr_d)/FLAGS.batch_num):
        batch_img=tr_d[j*FLAGS.batch_num:(j+1)*FLAGS.batch_num];
        batch_label=tr_l[j*FLAGS.batch_num:(j+1)*FLAGS.batch_num];
        # get adv imgs
        adv_img=sess.run([x_adv],feed_dict={img:np.concatenate([batch_img,batch_img],0),label:np.concatenate([batch_label,batch_label],0)})[0]
        for j in range(1,FLAGS.max_iter):
          adv_img=sess.run([x_adv],feed_dict={img:adv_img,label:np.concatenate([batch_label,batch_label],0)})[0];
        # training
        summary,tr_acc_org_b,tr_acc_adv_b,_=sess.run([merged,acc_org,acc_adv,trainer],feed_dict={img:np.concatenate([batch_img,adv_img[:FLAGS.batch_num]],0),label:np.concatenate([batch_label,batch_label],0)});
        tr_acc_org+=tr_acc_org_b*FLAGS.batch_num;
        tr_acc_adv+=tr_acc_adv_b*FLAGS.batch_num;
        #adv_img_set+=list(adv_img[:FLAGS.batch_num]);adv_img_set_label+=list(batch_label);
        #train_writer.add_summary(summary,i*len(tr_d)+j*FLAGS.batch_num);
      ts_acc=0;
      for j in range(len(ts_d)/(FLAGS.batch_num*2)):
        batch_img=ts_d[j*(FLAGS.batch_num*2):(j+1)*(FLAGS.batch_num*2)];
        batch_label=ts_l[j*(FLAGS.batch_num*2):(j+1)*(FLAGS.batch_num*2)];
        ts_acc_b1,ts_acc_b2=sess.run([acc_org,acc_adv],feed_dict={img:batch_img,label:batch_label});
        ts_acc+=(ts_acc_b1+ts_acc_b2)/2;
      # shuffle
      tr_idx=range(len(tr_d)); np.random.shuffle(tr_idx);
      tr_d=tr_d[tr_idx]; tr_l=tr_l[tr_idx];
      #adv_idx=range(len(adv_img_set));np.random.shuffle(adv_idx);
      #adv_img_set=np.array(adv_img_set);adv_img_set_label=np.array(adv_img_set_label);
      #adv_img_set=list(adv_img_set[adv_idx]);adv_img_set_label=list(adv_img_set_label[adv_idx]);
      #print(str(i+1)+" Epoch Training Acc_org/adv: "+str(tr_acc_org/len(tr_d))+"/"+str(tr_acc_adv/len(tr_d))+", Test Acc_org: "+str(ts_acc/(j+1)));
    saver.save(sess,os.path.join(FLAGS.save_folder,FLAGS.save_file_name));

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--max_epsilon', type=int, default=25,
                      help='max epsilon')
  parser.add_argument('--max_iter', type=int, default=1,
                      help='max iter on iterative GSM (If this one is 1, then FGSM)')
  parser.add_argument('--batch_num', type=int, default=1000,
                      help='the batch size')
  parser.add_argument('--max_epoch', type=int, default=100,
                      help='the max iteration')
  parser.add_argument('--tb_folder', type=str, default="/tmp/adv_training_cons/adv_training_fgsm",
                      help='the tensorboard log folder')
  parser.add_argument('--save_folder', type=str, default="/tmp/adv_training_cons/models/",
                      help='the model save folder')
  parser.add_argument('--save_file_name', type=str, default="adv_trained_fgsm.ckpt",
                      help='the model save file name')
  FLAGS, unparsed = parser.parse_known_args()
  train()

