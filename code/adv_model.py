import tensorflow as tf
import numpy as np
height=28;
width=28;
num_channel=3;
num_label=10;

def conv_variable(weight_shape,prefix):
  w = weight_shape[0]
  h = weight_shape[1]
  input_channels  = weight_shape[2]
  output_channels = weight_shape[3]
  d = 1.0 / np.sqrt(input_channels * w * h)
  bias_shape = [output_channels]
  weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d),name=prefix+"_w")
  bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d),name=prefix+"_b")
  return weight, bias

def conv2d(x, W, stride):
  return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def batch_norm_layer(x,training_phase,scope_bn,activation=None):
  return tf.cond(training_phase,
  lambda: tf.contrib.layers.batch_norm(x, activation_fn=activation, center=True, scale=True,
  updates_collections=None,is_training=True, reuse=None,scope=scope_bn,decay=0.9, epsilon=1e-5),
  lambda: tf.contrib.layers.batch_norm(x, activation_fn =activation, center=True, scale=True,
  updates_collections=None,is_training=False, reuse=True,scope=scope_bn,decay=0.9, epsilon=1e-5))

# Matching Nets
def MN(embedded_support_set,embedded_adv_imgs,labels,embedded_support_set_tf,embedded_adv_imgs_tf,embedded_support_set_label_tf,embedded_adv_imgs_label,sess,batch_num):
  adv_labels=np.zeros((len(embedded_adv_imgs),num_label),dtype=float);
  for i in range(len(labels)/batch_num):
    adv_labels[i*batch_num:(i+1)*batch_num]=sess.run(embedded_adv_imgs_label,feed_dict={embedded_support_set_tf:embedded_support_set,embedded_support_set_label_tf:labels,embedded_adv_imgs_tf:embedded_adv_imgs[i*batch_num:(i+1)*batch_num]});
  return adv_labels;

def MN_(num_support_set):
  embedded_imgs=tf.placeholder(tf.float32,[None,num_label],name="embedded_imgs");
  embedded_support_set=tf.placeholder(tf.float32,[num_support_set,num_label],name="embedded_support_set");
  embedded_support_set_label=tf.placeholder(tf.float32,[num_support_set,num_label],name="embedded_support_set_label");
  embedded_imgs_l2norm=tf.norm(embedded_imgs,axis=1);
  embedded_support_set_l2norm=tf.norm(embedded_support_set,axis=1);
  l2norm_set=tf.matmul(tf.reshape(embedded_imgs_l2norm,[-1,1]),tf.reshape(embedded_support_set_l2norm,[1,num_support_set]));
  embedded_dot_product=tf.matmul(embedded_imgs,tf.transpose(embedded_support_set));
  cosine_distance=tf.divide(embedded_dot_product,l2norm_set);
  exp_cosine_distance=tf.exp(cosine_distance);
  sum_exp_cosine_distance=tf.diag(tf.divide(1,tf.reduce_sum(exp_cosine_distance,axis=1)));
  labels=tf.matmul(tf.matmul(sum_exp_cosine_distance,exp_cosine_distance),embedded_support_set_label);
  return embedded_imgs,embedded_support_set,embedded_support_set_label,labels


# Just 3 layers Convnet
def classifier(img):
  w1,b1=conv_variable([3,3,num_channel,64],"c1");
  net=tf.nn.relu(conv2d(img,w1,2)+b1);
  w2,b2=conv_variable([3,3,64,128],"c2");
  net=tf.nn.relu(conv2d(net,w2,2)+b2);
  w3,b3=conv_variable([3,3,128,256],"c3");
  net=tf.nn.relu(conv2d(net,w3,2)+b3);
  net=tf.reshape(net,[-1,4096]);
  f_w1=tf.Variable(tf.truncated_normal([4096,8184],stddev=0.1),name="cf1_w");
  f_b1=tf.Variable(tf.constant(0.1,shape=[8184]),name="cf1_b");
  net=tf.nn.relu(tf.matmul(net,f_w1)+f_b1);
  f_w2=tf.Variable(tf.truncated_normal([8184,num_label],stddev=0.1),name="cf2_w");
  f_b2=tf.Variable(tf.constant(0.1,shape=[num_label]),name="cf2_b");
  net=tf.matmul(net,f_w2)+f_b2;
  prediction=tf.nn.softmax(net);

  params=[w1,b1,w2,b2,w3,b3,f_w1,f_b1,f_w2,f_b2];

  return params,net,prediction;

# Just 3 layers Convnet with kernel size as 5
def classifier2(img):
  w1,b1=conv_variable([5,5,num_channel,64],"c1");
  net=tf.nn.relu(conv2d(img,w1,2)+b1);
  w2,b2=conv_variable([5,5,64,128],"c2");
  net=tf.nn.relu(conv2d(net,w2,2)+b2);
  w3,b3=conv_variable([5,5,128,256],"c3");
  net=tf.nn.relu(conv2d(net,w3,2)+b3);
  net=tf.reshape(net,[-1,4096]);
  f_w1=tf.Variable(tf.truncated_normal([4096,8184],stddev=0.1),name="cf1_w");
  f_b1=tf.Variable(tf.constant(0.1,shape=[8184]),name="cf1_b");
  net=tf.nn.relu(tf.matmul(net,f_w1)+f_b1);
  f_w2=tf.Variable(tf.truncated_normal([8184,num_label],stddev=0.1),name="cf2_w");
  f_b2=tf.Variable(tf.constant(0.1,shape=[num_label]),name="cf2_b");
  net=tf.matmul(net,f_w2)+f_b2;
  prediction=tf.nn.softmax(net);

  params=[w1,b1,w2,b2,w3,b3,f_w1,f_b1,f_w2,f_b2];

  return params,net,prediction;

# Deeper model
def classifier3(img):
  w1_1,b1_1=conv_variable([3,3,num_channel,64],"c1_1");
  net=tf.nn.relu(conv2d(img,w1_1,2)+b1_1);
  w1_2,b1_2=conv_variable([3,3,64,64],"c1_2");
  net=tf.nn.relu(conv2d(net,w1_2,1)+b1_2);
  w1_3,b1_3=conv_variable([3,3,64,64],"c1_3");
  net=tf.nn.relu(conv2d(net,w1_3,1)+b1_3);
  w2_1,b2_1=conv_variable([3,3,64,128],"c2_1");
  net=tf.nn.relu(conv2d(net,w2_1,2)+b2_1);
  w2_2,b2_2=conv_variable([3,3,128,128],"c2_2");
  net=tf.nn.relu(conv2d(net,w2_2,1)+b2_2);
  w2_3,b2_3=conv_variable([3,3,128,128],"c2_3");
  net=tf.nn.relu(conv2d(net,w2_3,1)+b2_3);
  w3_1,b3_1=conv_variable([3,3,128,256],"c3_1");
  net=tf.nn.relu(conv2d(net,w3_1,2)+b3_1);
  w3_2,b3_2=conv_variable([3,3,256,256],"c3_2");
  net=tf.nn.relu(conv2d(net,w3_2,1)+b3_2);
  w3_3,b3_3=conv_variable([3,3,256,256],"c3_3");
  net=tf.nn.relu(conv2d(net,w3_3,1)+b3_3);
  net=tf.reshape(net,[-1,4096]);
  f_w1=tf.Variable(tf.truncated_normal([4096,8184],stddev=0.1),name="cf1_w");
  f_b1=tf.Variable(tf.constant(0.1,shape=[8184]),name="cf1_b");
  net=tf.nn.relu(tf.matmul(net,f_w1)+f_b1);
  f_w2=tf.Variable(tf.truncated_normal([8184,num_label],stddev=0.1),name="cf2_w");
  f_b2=tf.Variable(tf.constant(0.1,shape=[num_label]),name="cf2_b");
  net=tf.matmul(net,f_w2)+f_b2;
  prediction=tf.nn.softmax(net);

  params=[w1_1,b1_1,w1_2,b1_2,w1_3,b1_3,w2_1,b2_1,w2_2,b2_2,w2_3,b2_3,w3_1,b3_1,w3_2,b3_2,w3_3,b3_3,f_w1,f_b1,f_w2,f_b2];

  return params,net,prediction;

# Resnet model
def classifier4(img):
  w1_1,b1_1=conv_variable([3,3,num_channel,64],"c1_1");
  net=tf.nn.relu(conv2d(img,w1_1,2)+b1_1);
  w1_2,b1_2=conv_variable([3,3,64,64],"c1_2");
  net=tf.nn.relu(conv2d(net,w1_2,1)+b1_2+net);
  w1_3,b1_3=conv_variable([3,3,64,64],"c1_3");
  net=tf.nn.relu(conv2d(net,w1_3,1)+b1_3+net);
  w2_1,b2_1=conv_variable([3,3,64,128],"c2_1");
  net=tf.nn.relu(conv2d(net,w2_1,2)+b2_1);
  w2_2,b2_2=conv_variable([3,3,128,128],"c2_2");
  net=tf.nn.relu(conv2d(net,w2_2,1)+b2_2+net);
  w2_3,b2_3=conv_variable([3,3,128,128],"c2_3");
  net=tf.nn.relu(conv2d(net,w2_3,1)+b2_3+net);
  w3_1,b3_1=conv_variable([3,3,128,256],"c3_1");
  net=tf.nn.relu(conv2d(net,w3_1,2)+b3_1);
  w3_2,b3_2=conv_variable([3,3,256,256],"c3_2");
  net=tf.nn.relu(conv2d(net,w3_2,1)+b3_2+net);
  w3_3,b3_3=conv_variable([3,3,256,256],"c3_3");
  net=tf.nn.relu(conv2d(net,w3_3,1)+b3_3+net);
  net=tf.reshape(net,[-1,4096]);
  f_w1=tf.Variable(tf.truncated_normal([4096,8184],stddev=0.1),name="cf1_w");
  f_b1=tf.Variable(tf.constant(0.1,shape=[8184]),name="cf1_b");
  net=tf.nn.relu(tf.matmul(net,f_w1)+f_b1);
  f_w2=tf.Variable(tf.truncated_normal([8184,num_label],stddev=0.1),name="cf2_w");
  f_b2=tf.Variable(tf.constant(0.1,shape=[num_label]),name="cf2_b");
  net=tf.matmul(net,f_w2)+f_b2;
  prediction=tf.nn.softmax(net);

  params=[w1_1,b1_1,w1_2,b1_2,w1_3,b1_3,w2_1,b2_1,w2_2,b2_2,w2_3,b2_3,w3_1,b3_1,w3_2,b3_2,w3_3,b3_3,f_w1,f_b1,f_w2,f_b2];

  return params,net,prediction;

def attacker(x_data,is_training,batch_num):
    # Small epsilon value for the BN transform
    epsilon = 1e-3;
    # Reference model : Learning Deconvlutional Network for Semantic Segmentation
    #conv_layer maximum is 5
    conv_layer_num=3;
    fil_num_list=[32,64,128,256,512];
    #fil_num_list=[64,128,256,512,1024];
    #fil_num_list=[64,64,64,64,64,64,64];
    c_W=np.zeros(conv_layer_num,dtype=object);
    c_b=np.zeros(conv_layer_num,dtype=object);
    conv=np.zeros(conv_layer_num,dtype=object);
    mean_variance_var_list=[];
    input_data = batch_norm_layer(x_data,training_phase=is_training,scope_bn='gene_bn_input',activation=tf.identity)
    mean_variance_var_list+=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='gene_bn_input');
    # conv layers
    c_W[0] = tf.Variable(tf.truncated_normal([3, 3, num_channel, fil_num_list[0]], stddev=0.1),name="a0_cw")
    c_b[0] = tf.Variable(tf.constant(0.1, shape=[fil_num_list[0]]),name="a0_cb")
    conv_res = tf.nn.conv2d(input_data, c_W[0], strides=[1, 2, 2, 1], padding='SAME')+c_b[0];
    conv[0] = batch_norm_layer(conv_res,training_phase=is_training,scope_bn='gene_bn_conv_0',activation=tf.nn.relu)
    mean_variance_var_list+=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='gene_bn_conv_0');
    for i in range(1,conv_layer_num):
      c_W[i] = tf.Variable(tf.truncated_normal([3, 3, fil_num_list[i-1], fil_num_list[i]], stddev=0.1),name="a"+str(i)+"_cw")
      c_b[i] = tf.Variable(tf.constant(0.1, shape=[fil_num_list[i]]),name="a"+str(i)+"_cb")
      conv_res = tf.nn.conv2d(conv[i-1], c_W[i], strides=[1, 2, 2, 1], padding='SAME')+c_b[i];
      conv[i] = batch_norm_layer(conv_res,training_phase=is_training,scope_bn='gene_bn_conv_'+str(i),activation=tf.nn.relu)
      mean_variance_var_list+=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='gene_bn_conv_'+str(i));
    # fc layer
    f_w = tf.Variable(tf.truncated_normal([1, 1, fil_num_list[conv_layer_num-1], fil_num_list[conv_layer_num-1]], stddev=0.1),name="a_fw")
    f_b = tf.Variable(tf.constant(0.1, shape=[fil_num_list[conv_layer_num-1]]),name="a_fb")
    net_res = tf.nn.conv2d(conv[conv_layer_num-1], f_w, strides=[1, 1, 1, 1], padding='SAME')+f_b;
    net = batch_norm_layer(net_res,training_phase=is_training,scope_bn='gene_bn_fc',activation=tf.nn.relu)
    mean_variance_var_list+=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='gene_bn_fc');
    # deconv layers
    d_W=np.zeros(conv_layer_num,dtype=object);
    d_b=np.zeros(conv_layer_num,dtype=object);
    for i in range(1,conv_layer_num):
      idx=conv_layer_num-i;
      d_W[idx] = tf.Variable(tf.truncated_normal([3, 3, fil_num_list[idx-1], fil_num_list[idx]], stddev=0.1),name="a"+str(idx)+"_dw")
      d_b[idx] = tf.Variable(tf.constant(0.1, shape=[fil_num_list[idx-1]]),name="a"+str(idx)+"_db")
      x_shape = tf.shape(conv[idx-1]);
      out_shape = tf.stack([x_shape[0], x_shape[1], x_shape[2], fil_num_list[idx-1]])
      deconv_res = tf.nn.conv2d_transpose(net, d_W[idx], output_shape=out_shape ,strides=[1, 2, 2, 1], padding='SAME')+d_b[idx]
      net = batch_norm_layer(deconv_res,training_phase=is_training,scope_bn='gene_bn_deconv_'+str(idx),activation=tf.nn.relu)
      mean_variance_var_list+=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='gene_bn_deconv_'+str(idx));
    d_W[0] = tf.Variable(tf.truncated_normal([3, 3, num_channel, fil_num_list[0]], stddev=0.1),name="a0_dw")
    d_b[0] = tf.Variable(tf.constant(0.1, shape=[num_channel]),name="a0_db")
    net = tf.nn.conv2d_transpose(net, d_W[0], output_shape=[batch_num,height,width,num_channel],strides=[1, 2, 2, 1], padding='SAME')+d_b[0];
    x_generate = batch_norm_layer(net,training_phase=is_training,scope_bn='gene_bn_deconv_0',activation=tf.nn.tanh)
    mean_variance_var_list+=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='gene_bn_deconv_0');

    bn_var_num=len(mean_variance_var_list);
    g_params=list(c_W)+list(c_b)+[f_w,f_b]+list(d_W)+list(d_b)+mean_variance_var_list;

    return x_generate, g_params

