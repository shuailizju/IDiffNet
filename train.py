""" This script demonstrates the use of a convolutional LSTM network.
This network is used to predict the next frame of an artificially
generated movie which contains moving squares.
"""


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import numpy as np
#import pylab as plt
import h5py
import tensorflow as tf
import math
#import net_conv_deconv_diffuser
import densenet_diffuserL2_equal
import keras.backend as K
from keras import optimizers, metrics
from keras.optimizers import RMSprop, Adam, Nadam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, Callback, CSVLogger
from keras.layers import Permute
from keras.models import load_model
#from keras.utils.visualize_util import plot
from keras.utils.io_utils import HDF5Matrix




def run_experiment(case_data, typnet="res", u_net=True, sh_add=4, ty_2=True, input_path='F:\\Optica_data\\',dataname='TIE_iter_face.mat',datanamePh='F:\\Optica_data\\',dataname2='TIE_iter_face.mat',dataname3='TIE_iter_face.mat', dataname4='TIE_iter_face.mat', weight_save_path='F:\\Optica_data\\',fileend1=42500,filestr=42000,fileend2=45000,imginput=True,normz=1):

        # learning rate schedule


         
        def step_decay(epoch):
                initial_lrate = 0.001
                drop = 0.5
                epochs_drop = 5.0
                lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
                return lrate

        class LossHistory():
            def on_train_begin(self, logs={}):
                self.losses = []

            def on_batch_end(self, batch, logs={}):
                self.losses.append(logs.get('loss'))


        def normalize_data1(x):
                #x=phase_matlin[np.array(x)]
                x=x*100
                return x

        def normalize_data2(x):
                #x=phase_mattri[np.array(x)]
                x=x*100
                return x

        if case_data==1:  
                file_base="Face_"
        if case_data==2:               
                file_base="Digits_"
        if case_data==3:          
                file_base="Images_"


        
        if normz==0:
                output_mat = HDF5Matrix(input_path+dataname, 'OrgMat1',0, fileend1, normalizer=None)
                output_mat_test = HDF5Matrix(input_path+dataname2, 'OrgMat1',filestr, fileend2, normalizer=None)
        elif normz==1:
                output_mat = HDF5Matrix(input_path+dataname, 'OrgMat1',0, fileend1, normalizer=normalize_data1)
                output_mat_test = HDF5Matrix(input_path+dataname2, 'OrgMat1',filestr, fileend2, normalizer=normalize_data1)
        elif normz==2:
                output_mat = HDF5Matrix(input_path+dataname, 'OrgMat1',0, fileend1, normalizer=normalize_data2)
                output_mat_test = HDF5Matrix(input_path+dataname2, 'OrgMat1',filestr, fileend2, normalizer=normalize_data2)
                
        input_mat = HDF5Matrix(input_path+dataname3, 'CamMat_denoise',0, fileend1, normalizer=None)
        input_mat_test = HDF5Matrix(input_path+dataname4, 'CamMat_denoise',filestr, fileend2, normalizer=None)

              
     


        #test
        


        shape_aux= (1,128,128)

        if typnet=="dense":
                #depth = 40
                #depth = 40
                nb_dense_block = 6
                nb_layers_per_block = 3
                growth_rate = 12 #16
                nb_filter = 16 #24
                dropout_rate = 0.05 
                #bottleneck=False
                reduction=0.5
                nb_classes=1
                include_top = True
                #, upsampling_type='subpixel'
                model = densenet_diffuserL2_equal.DenseNetFCN(shape_aux, classes=nb_classes,  nb_dense_block=nb_dense_block, nb_layers_per_block=nb_layers_per_block,
                          growth_rate=growth_rate, init_conv_filters=nb_filter, dropout_rate=dropout_rate, reduction=reduction, include_top = include_top,upsampling_type='subpixel')
                model.summary()
                #model = net_conv_deconv_Type1_optica.ResnetBuilder.build_plainnet(shape_aux, type2=ty_2, shape_add=sh_add, unet=u_net, imginput=imginput)
                batch_sz=8
        else:
                if normz==0:
                        model = net_conv_deconv_diffuser.ResnetBuilder.build_resnet(shape_aux, type2=ty_2, shape_add=sh_add, unet=u_net, imginput=imginput)
                elif normz==1:
                        model = net_conv_deconv_diffuser.ResnetBuilder.build_resnet(shape_aux, type2=ty_2, shape_add=sh_add, unet=u_net, imginput=imginput)
                elif normz==2:
                        model = net_conv_deconv_diffuser.ResnetBuilder.build_resnet(shape_aux, type2=ty_2, shape_add=sh_add, unet=u_net, imginput=imginput)
                batch_sz=3

        if not os.path.exists(weight_save_path+dataname[:-4]):
                os.makedirs(weight_save_path+dataname[:-4])
        fileall= weight_save_path+dataname[:-4]+"\\Diffuser_densenature_"+file_base+typnet+"_type2"+str(ty_2)+"_shapeadd"+str(sh_add)+"_unet"+ str(u_net)+"_imageinpit"+ str(imginput)+"_normz"+str(normz)
        filepath=fileall+"_weights_type2.{epoch:02d}.hdf5"       
        filename=fileall+'training.log'

        #densealtsubskip43L is 53L

        


 #       adam=optimizers.RMSprop(clipvalue=5)
        adam=optimizers.Adam(clipvalue=1)
        #clipvalue=1
        model.compile(loss='pearson_cc_1', optimizer=adam, metrics=[metrics.mae, metrics.pearson_cc_1])
        int_eph=0

        if dataname=='F:\\Diffuser\\Imagesdiffus5er_rotation220R_optica_distance4400_1.mat':
                model.load_weights('F:\Diffuser\Imagesdiffuser_rotation220R_optica_distance4400_1\Diffuser_Images_res_type2False_shapeadd512_unetFalse_imageinpitTrue_normz0_weights_type2.02.hdf5')
                int_eph=3
        

        
        csv_logger = CSVLogger(filename)
        lrate = LearningRateScheduler(step_decay)
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')

        callbacks_list = [lrate, checkpoint,csv_logger]
        #callbacks_list = [checkpoint,csv_logger]

        
        if ty_2 and not imginput:
                model.fit([input_mat,idx_mat], output_mat, batch_size=batch_sz,
                nb_epoch=20, validation_data=([input_mat_test,idx_mat_test], output_mat_test), callbacks=callbacks_list,  shuffle="batch")
        elif ty_2 and imginput:
                model.fit([input_mat,holo_mat], output_mat, batch_size=batch_sz,
                nb_epoch=20, validation_data=([input_mat_test,holo_mat_test], output_mat_test), callbacks=callbacks_list,  shuffle="batch")
        else:
                model.fit(input_mat, output_mat, batch_size=batch_sz,
                nb_epoch=20, validation_data=(input_mat_test, output_mat_test), callbacks=callbacks_list,  shuffle="batch",initial_epoch=int_eph)


if __name__ == '__main__':

    base_folder='C:\\Users\\Shuai\\Dropbox (MIT)\\Shuai Li lab1\\Diffuser_Nature\\'
    input_folder='Raw_data_static_128_512_conversion_resolution_grit400_exp30000\\'
    weight_save_folder='Static_128_512_weights_resolution\\'
    subname='diffuser_rot'
    dist=46800
    grit=400
    normtype=0


 #   run_experiment(case_data=1, typnet="dense", u_net=False, sh_add=512, ty_2=False,input_path=base_folder+input_folder,dataname='Faces'+subname+'_optica_distance'+str(dist)+'_'+str(grit)+'_c.mat',datanamePh=base_folder,
#                  dataname2='Test'+subname+'_optica_distance'+str(dist)+'_'+str(grit)+'_c.mat', dataname3='Faces'+subname+'_optica_distance'+str(dist)+'_'+str(grit)+'_denoise128_c.mat',
#                 dataname4='Test'+subname+'_optica_distance'+str(dist)+'_'+str(grit)+'_denoise128_c.mat', weight_save_path=base_folder+weight_save_folder,fileend1=10000,filestr=250, fileend2=350,imginput=True,normz=normtype)


    run_experiment(case_data=2, typnet="dense", u_net=False, sh_add=512, ty_2=False,input_path=base_folder+input_folder,dataname='Digits'+subname+'_optica_distance'+str(dist)+'_'+str(grit)+'_c.mat',datanamePh=base_folder,
                 dataname2='Test'+subname+'_optica_distance'+str(dist)+'_'+str(grit)+'_c.mat',dataname3='Digits'+subname+'_optica_distance'+str(dist)+'_'+str(grit)+'_denoise128_c.mat',
                  dataname4='Test'+subname+'_optica_distance'+str(dist)+'_'+str(grit)+'_denoise128_c.mat', weight_save_path=base_folder+weight_save_folder,fileend1=10000,filestr=150, fileend2=250,imginput=True,normz=normtype)

 #   run_experiment(case_data=3, typnet="dense", u_net=False, sh_add=512, ty_2=False,input_path=base_folder+input_folder,dataname='Images'+subname+'_optica_distance'+str(dist)+'_'+str(grit)+'_c.mat',datanamePh=base_folder,
#                  dataname2='Test'+subname+'_optica_distance'+str(dist)+'_'+str(grit)+'_c.mat', dataname3='Images'+subname+'_optica_distance'+str(dist)+'_'+str(grit)+'_denoise128_c.mat',
#                 dataname4='Test'+subname+'_optica_distance'+str(dist)+'_'+str(grit)+'_denoise128_c.mat', weight_save_path=base_folder+weight_save_folder,fileend1=10000,filestr=350, fileend2=450,imginput=True,normz=normtype)






    
    


