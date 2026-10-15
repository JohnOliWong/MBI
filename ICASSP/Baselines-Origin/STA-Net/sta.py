import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

# If you have any questions, please contact us at shulmt@shu.edu.cn

# using pearson correlation coefficients to determine the similarity between eeg and fnirs
def pearson_r(eeg, fnirs):
    mx = tf.math.reduce_mean(eeg, axis=1, keepdims=True)
    my = tf.math.reduce_mean(fnirs, axis=1, keepdims=True)
    xm, ym = eeg-mx, fnirs-my
    r_num = tf.math.reduce_mean(tf.multiply(xm,ym), axis=1)        
    r_den = tf.math.reduce_std(xm, axis=1) * tf.math.reduce_std(ym, axis=1) + 1e-6
    plcc = r_num / r_den
    plcc = tf.math.abs(plcc)
    plcc_meanbatch = tf.math.reduce_mean(plcc)
    
    return plcc_meanbatch


class pos_embedding(layers.Layer):
    def __init__(self):
        super(pos_embedding, self).__init__()

    def build(self, input_shape):
        self.pos_embedding = self.add_weight(name='pos_embedding', 
                        shape=(1, input_shape[-2], input_shape[-1]),
                        initializer=tf.keras.initializers.HeUniform(),
                        trainable=True)
        
    def call(self, inputs):
        return inputs + self.pos_embedding
    

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

    
class e_f_attention(keras.layers.Layer):
    def __init__(self, emb_size, d_model, heads, drop):
        super(e_f_attention, self).__init__()

        self.q_flat = layers.Flatten()
        self.q_proj = layers.Dense(emb_size)

        self.fusion_proj = layers.Dense(emb_size)

        self.k_flat = layers.Reshape((11, -1))
        self.k_proj = layers.Dense(emb_size)
        self.pos = pos_embedding()

        self.dot_product_attention = layers.MultiHeadAttention(num_heads=heads, key_dim=d_model, dropout=drop)

        self.ef_plcc_tracker = keras.metrics.Mean(name="ef_plcc")

    def call(self, inputs):
        eeg, fnirs = inputs

        q_eeg = self.q_flat(eeg)

        fusion_output = self.fusion_proj(q_eeg)

        q_eeg = self.q_proj(q_eeg)
        q_eeg = tf.expand_dims(q_eeg, axis=1) 

        k_fnirs = self.k_flat(fnirs)
        k_fnirs = self.pos(k_fnirs)
        k_fnirs = self.k_proj(k_fnirs) 
        
        fnirs_weighted, attention_weights = self.dot_product_attention(q_eeg, k_fnirs, return_attention_scores=True) 
        attention_weights = tf.math.reduce_mean(attention_weights, axis=(1, 2))
        
        q_eeg = tf.math.reduce_mean(q_eeg, axis=1)
        fnirs_weighted = tf.math.reduce_mean(fnirs_weighted, axis=1)

        ef_loss = pearson_r(q_eeg, fnirs_weighted)
        
        self.add_loss(1-ef_loss)

        self.ef_plcc_tracker.update_state(ef_loss)
       
        return fusion_output, fnirs_weighted, attention_weights
    

class gap(keras.layers.Layer):
    def __init__(self):
        super(gap, self).__init__()

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=-2, keepdims=True)
    

class fga(keras.layers.Layer):
    def __init__(self, tem_kernel_size, fga_loss_name):
        super(fga, self).__init__()

        self.channel_pooling = layers.Conv3D(filters=1, kernel_size=(3, 3, tem_kernel_size), strides=(1, 1, 1), padding='same')

        self.tap_fnirs = gap()

        self.residual_para = self.add_weight(name='residual_para', initializer="zeros", trainable=True)

        self.add_eeg = layers.Add()
        self.add = layers.Add()

        self.eeg_flatten = layers.Flatten()
        self.fnirs_flatten = layers.Flatten()
        self.fusion_flatten = layers.Flatten()

        self.fga_loss_tracker = keras.metrics.Mean(name=fga_loss_name)

    def call(self, inputs):
        eeg_fusion, eeg, fnirs = inputs
        
        fnirs_attention = self.channel_pooling(fnirs) 

        fnirs_attention_map = self.tap_fnirs(fnirs_attention) 
        fnirs_attention_map = tf.math.reduce_mean(fnirs_attention_map, axis=1) 
        
        fnirs_attention_map_norm = keras.activations.sigmoid(fnirs_attention_map) 

        eeg_fusion_guided = tf.math.multiply(eeg_fusion, fnirs_attention_map_norm) 

        residual_para_norm = keras.activations.sigmoid(self.residual_para)
        eeg_add = self.add_eeg([residual_para_norm*eeg, (1-residual_para_norm)*eeg_fusion]) 

        fga_feature = self.add([eeg_fusion_guided, eeg_add]) 


        eeg_plcc = tf.math.reduce_mean(eeg, axis=(-1, -2)) 
        eeg_plcc = self.eeg_flatten(eeg_plcc) 
        
        fnirs_attention_map_norm_plcc = self.fnirs_flatten(fnirs_attention_map_norm) 

        fga_loss = pearson_r(eeg_plcc, fnirs_attention_map_norm_plcc)

        self.add_loss(1-fga_loss)

        self.fga_loss_tracker.update_state(fga_loss)
        

        return fga_feature
    

class conv_block(keras.layers.Layer):
    def __init__(self, eeg_filter, eeg_size, eeg_stride,
                fnirs_filter, fnirs_size, fnirs_stride,
                eegfusion_filter, eegfusion_size, eegfusion_stride, 
                tem_kernel_size, fga_loss_name, padding):
        super(conv_block, self).__init__()

        self.eeg_conv = layers.Conv3D(filters=eeg_filter, kernel_size=eeg_size, strides=eeg_stride, padding=padding)
        self.eeg_act = layers.Activation('elu')
        self.eeg_bn = layers.BatchNormalization()

        self.fnirs_conv = layers.Conv3D(filters=fnirs_filter, kernel_size=fnirs_size, strides=fnirs_stride, padding=padding)
        self.fnirs_act = layers.Activation('elu')
        self.fnirs_bn = layers.BatchNormalization()

        self.eegfusion_conv = layers.Conv3D(filters=eegfusion_filter, kernel_size=eegfusion_size, strides=eegfusion_stride, padding=padding)
        self.eegfusion_act = layers.Activation('elu')
        self.eegfusion_bn = layers.BatchNormalization()

        self.fga = fga(tem_kernel_size, fga_loss_name)

    def call(self, inputs):
        eegfusion, eeg, fnirs = inputs

        eeg_feature = self.eeg_conv(eeg) 
        eeg_feature = self.eeg_bn(eeg_feature) 
        eeg_feature = self.eeg_act(eeg_feature) 

        fnirs_feature = self.fnirs_conv(fnirs) 
        fnirs_feature = self.fnirs_bn(fnirs_feature) 
        fnirs_feature = self.fnirs_act(fnirs_feature) 

        eegfusion_feature = self.eegfusion_conv(eegfusion) 
        eegfusion_feature = self.eegfusion_bn(eegfusion_feature) 
        eegfusion_feature = self.eegfusion_act(eegfusion_feature) 

        eegfusion_fga = self.fga((eegfusion_feature, eeg_feature, fnirs_feature)) 

        return eegfusion_fga, eeg_feature, fnirs_feature


# class conv_block(layers.Layer):
#     def __init__(self, eeg_filter, eeg_size, eeg_stride,
#                 fnirs_filter, fnirs_size, fnirs_stride,
#                 eegfusion_filter, eegfusion_size, eegfusion_stride, 
#                 tem_kernel_size, fga_loss_name, padding, **kwargs):
#         super(conv_block, self).__init__(**kwargs)
#         # 参数保存（略，同之前代码）

#     def build(self, input_shape):
#         # 处理可能的列表输入（多分支场景）
#         if isinstance(input_shape, list):
#             input_shape = input_shape[0]  # 假设所有分支形状相同
        
#         # 初始化子层
#         self.eeg_conv = layers.Conv3D(
#             filters=self.eeg_filter, kernel_size=self.eeg_size,
#             strides=self.eeg_stride, padding=self.padding
#         )
#         self.eeg_conv.build(input_shape)
        
#         self.eeg_bn = layers.BatchNormalization()
#         self.eeg_bn.build(input_shape)  # 需要明确的通道轴（通常是最后一个维度）
        
#         # 同理初始化其他子层...
        
#         # 初始化 fga 层
#         self.fga_layer = fga(self.tem_kernel_size, self.fga_loss_name)
#         self.fga_layer.build([input_shape] * 3)  # 传递 3 个相同的形状
        
#         super(conv_block, self).build(input_shape)

#     def call(self, inputs, training=None):
#         # EEG 分支
#         x_eeg = self.eeg_conv(inputs)
#         x_eeg = self.eeg_act(x_eeg)
#         x_eeg = self.eeg_bn(x_eeg, training=training)

#         # fNIRS 分支
#         x_fnirs = self.fnirs_conv(inputs)
#         x_fnirs = self.fnirs_act(x_fnirs)
#         x_fnirs = self.fnirs_bn(x_fnirs, training=training)

#         # EEG-Fusion 分支
#         x_eegfusion = self.eegfusion_conv(inputs)
#         x_eegfusion = self.eegfusion_act(x_eegfusion)
#         x_eegfusion = self.eegfusion_bn(x_eegfusion, training=training)

#         # 应用 FGA
#         output = self.fga_layer([x_eeg, x_fnirs, x_eegfusion])
#         return output

#     def get_config(self):
#         # 支持序列化
#         config = super(conv_block, self).get_config()
#         config.update({
#             'eeg_filter': self.eeg_filter,
#             'eeg_size': self.eeg_size,
#             'eeg_stride': self.eeg_stride,
#             'fnirs_filter': self.fnirs_filter,
#             'fnirs_size': self.fnirs_size,
#             'fnirs_stride': self.fnirs_stride,
#             'eegfusion_filter': self.eegfusion_filter,
#             'eegfusion_size': self.eegfusion_size,
#             'eegfusion_stride': self.eegfusion_stride,
#             'tem_kernel_size': self.tem_kernel_size,
#             'fga_loss_name': self.fga_loss_name,
#             'padding': self.padding
#         })
#         return config


class reduce_sum_layer(keras.layers.Layer):
    def __init__(self, axis, keepaxis, name=None, *args, **kwargs):
        super(reduce_sum_layer, self).__init__(name=name)

        self.axis = axis
        self.keepaxis = keepaxis

    def call(self, inputs):
        return tf.math.reduce_sum(inputs, axis=self.axis, keepdims=self.keepaxis)
    

class reduce_mean_layer(keras.layers.Layer):
    def __init__(self, axis, keepaxis, name=None, *args, **kwargs):
        super(reduce_mean_layer, self).__init__(name=name)

        self.axis = axis
        self.keepaxis = keepaxis

    def call(self, inputs):
        return tf.math.reduce_mean(inputs, axis=self.axis, keepdims=self.keepaxis)
    

class expand_dims_layer(keras.layers.Layer):
    def __init__(self, axis, *args, **kwargs):
        super(expand_dims_layer, self).__init__()

        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)
    

class prediction_weight_layer(keras.layers.Layer):
    def __init__(self, name=None):
        super(prediction_weight_layer, self).__init__(name=name)

        self.p_weight = self.add_weight(name='p_weight', shape=(2,1), initializer="zeros", trainable=True)

    def call(self, inputs):
        p_weights_softmax = tf.nn.softmax(self.p_weight, axis=0) 

        p_weights_softmax = tf.expand_dims(p_weights_softmax, axis=0) 
        
        the_prediction = tf.math.multiply(inputs, p_weights_softmax) 
        the_prediction = tf.math.reduce_sum(the_prediction, axis=1) 

        return the_prediction
        

def sta_net():
    eeg_input = keras.Input(shape=(16, 16, 600, 1), name="eeg_input")
    fnirs_input = keras.Input(shape=(11, 16, 16, 30, 2), name="fnirs_input")
    print(eeg_input.shape, fnirs_input.shape)

    eegfusion1, eeg1, fnirs1 = conv_block(eeg_filter=16, eeg_size=(2, 2, 13), eeg_stride=(2, 2, 6),
                                          fnirs_filter=16, fnirs_size=(2, 2, 5), fnirs_stride=(2, 2, 2),
                                          eegfusion_filter=16, eegfusion_size=(2, 2, 13), eegfusion_stride=(2, 2, 6),
                                          tem_kernel_size=5, fga_loss_name='fgsa1_plcc', padding='same')((eeg_input, eeg_input, fnirs_input))
    eegfusion1 = layers.Dropout(0.5)(eegfusion1) 
    eeg1 = layers.Dropout(0.5)(eeg1)
    fnirs1 = layers.Dropout(0.5)(fnirs1)                            

    eegfusion2, eeg2, fnirs2 = conv_block(eeg_filter=32, eeg_size=(2, 2, 5), eeg_stride=(2, 2, 2),
                                          fnirs_filter=32, fnirs_size=(2, 2, 3), fnirs_stride=(2, 2, 2),
                                          eegfusion_filter=32, eegfusion_size=(2, 2, 5), eegfusion_stride=(2, 2, 2),
                                          tem_kernel_size=3, fga_loss_name='fgsa2_plcc', padding='same')((eegfusion1, eeg1, fnirs1)) 
    eegfusion2 = gap()(eegfusion2)
    eeg2 = gap()(eeg2)
    fnirs2 = gap()(fnirs2)

    eegfusion2 = layers.Dropout(0.5)(eegfusion2)
    eeg2 = layers.Dropout(0.5)(eeg2)
    fnirs2 = layers.Dropout(0.5)(fnirs2)

    eegfusion_feature, fnirs_feature, _ = e_f_attention(emb_size=256, d_model=256, heads=10, drop=0.5)((eegfusion2, fnirs2))
    eegfusion_feature_pweight = layers.Activation('elu')(eegfusion_feature)
    fnirs_feature_pweight = layers.Activation('elu')(fnirs_feature)

    eegfusion_feature_pweight = layers.Dense(256, activation='elu')(eegfusion_feature_pweight)
    fnirs_feature_pweight = layers.Dense(256, activation='elu')(fnirs_feature_pweight)

    eeg_feature = layers.Flatten()(eeg2)
    eeg_feature = layers.Dense(256, activation='elu')(eeg_feature)
    
    eegfusion_pred = layers.Dense(2)(eegfusion_feature_pweight)
    fnirs_pred = layers.Dense(2)(fnirs_feature_pweight)
    eeg_pred = layers.Dense(2)(eeg_feature)

    eeg_pred = layers.Activation('softmax', name='eeg_output')(eeg_pred) 
    eegfusion_pred = layers.Activation('softmax')(eegfusion_pred) 
    fnirs_pred = layers.Activation('softmax')(fnirs_pred) 
    
    eegfusion_pred = expand_dims_layer(axis=1)(eegfusion_pred) 
    fnirs_pred = expand_dims_layer(axis=1)(fnirs_pred) 

    the_pred = layers.Concatenate(axis=1)([eegfusion_pred, fnirs_pred]) 
    
    fnirs_p_weight = layers.Dense(1)(fnirs_feature_pweight) 
    eegfusion_p_weight = layers.Dense(1)(eegfusion_feature_pweight) 
    
    p_weight = layers.Concatenate()([eegfusion_p_weight, fnirs_p_weight]) 
    p_weight = layers.Activation('softmax')(p_weight) 
    p_weight = expand_dims_layer(axis=-1)(p_weight) 

    the_pred = layers.Multiply()([the_pred, p_weight]) 
    the_pred = reduce_sum_layer(axis=1, keepaxis=False, name='class_output')(the_pred) 
    
    # returns a defined model
    model = keras.Model(inputs=[eeg_input, fnirs_input], outputs=[the_pred, eeg_pred])

    return model