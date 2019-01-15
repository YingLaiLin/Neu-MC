import tensorflow as tf
import keras
from keras import backend as K
from keras import regularizers
from keras import layers
import numpy as np
import h5py
from keras.models import Model
from keras.layers import Embedding, Input, Dense, Activation, dot, Layer
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Dropout,BatchNormalization, Add, multiply
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
import argparse
import scipy.io as sio
from Dataset import Dataset
import matlab.engine
from time import time
from evaluate import evaluate_model
import pandas as pd
import os
import pygpu
from keras.constraints import UnitNorm


def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
            help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='gene',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--proj_dim', type=int, default=500,
                        help='Projection size of gene and disease.')
    parser.add_argument('--r', type=int, default=100,
                        help='specify Top K for evaluation.')                
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='weight of unlabled observatons in loss function')
    parser.add_argument('--reg_gene', type=float, default=5e-4,
                        help='Regularization for gene projection.')                    
    parser.add_argument('--reg_disease', type=float, default=5e-4,
                        help='Regularization for disease projection.')                        
    parser.add_argument('--num_neg', type=int, default=10,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--decay', type=float, default=6e-3,
                        help='Decay rate for learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--drop', type=float, default=0.1,
                        help='Drop rate for dropout.')
    parser.add_argument('--verbose', type=int, default=50,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--model', nargs='?', default='nimc', 
                        help='Specify the model to use: nimc, deepimc')
    parser.add_argument('--pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for this model.')
    return parser.parse_args()

def init_weights(shape, name=None):
    feas = h5py.File('projection.mat','r')
    projection = feas['proj'][:,:]
    return K.variable(projection)

def residual_block(x, reg, proj_dim, bottle_neck_dim):
    y = Dense(proj_dim,trainable=True, kernel_regularizer=regularizers.l2(5e-4),
            activation='relu', kernel_initializer='random_normal')(x)
    z = Dense(proj_dim,trainable=True, kernel_regularizer=regularizers.l2(5e-4),
            activation='relu', kernel_initializer='random_normal')(y)
    return layers.add([x, z])

def get_nimc_model(num_genes, num_diseases, proj_dim, reg_gene, reg_disease):
    # Input variables
    gene_input = Input(shape=(1,), dtype='int32', name='gene_input')
    disease_input = Input(shape=(1,), dtype='int32', name='disease_input')

    # using gene feature matrix for initialization
    gene_feas = h5py.File('gene_features.mat', 'r')
    humannet_features = gene_feas['features'][:,:].T
    gene_feature_size = humannet_features.shape
    assert gene_feature_size[0] + 1 == num_genes

    humannet_features = np.insert(humannet_features, 0, [0]*gene_feature_size[1], 0)
    #humannet_embedding_layer = Embedding(input_dim=gene_feature_size[0]+1, output_dim=gene_feature_size[1], trainable=False)
    #humannet_embedding_layer.build((None,))
    #humannet_embedding_layer.set_weights([humannet_features])
    humannet_embedding_layer = Embedding(input_dim=gene_feature_size[0]+1, output_dim=gene_feature_size[1], trainable=False, weights=[humannet_features])
    # using disease feature matrix for initialization
    feas = h5py.File('disease_features.mat','r')
    omim_features = feas['col_features'][:,:].T
    disease_feature_size = omim_features.shape
    assert disease_feature_size[0] + 1 == num_diseases
    
    omim_features = np.insert(omim_features, 0, [0]*disease_feature_size[1], 0)
    #omim_embedding_layer = Embedding(input_dim=disease_feature_size[0]+1, output_dim=disease_feature_size[1],trainable=False)
    #omim_embedding_layer.build((None,))
    #omim_embedding_layer.set_weights([omim_features])
    omim_embedding_layer = Embedding(input_dim=disease_feature_size[0]+1, output_dim=disease_feature_size[1],trainable=False,weights=[omim_features])
    # get specific features
    gene_feature = Flatten()(humannet_embedding_layer(gene_input))
    disease_feature = Flatten()(omim_embedding_layer(disease_input))
    # disease_feature = K.transpose(disease_feature)
    # projection matrix, using W * H' as initialization
    

    # projection of gene feature and disease feature
    
    
    
    projected_gene_feature = Dense(proj_dim,trainable=True, 
                kernel_regularizer=regularizers.l2(reg_gene), name='gene_projection',
                activation='relu', kernel_initializer='he_normal')(gene_feature)

    #compact_gene_feature = residual_block(residual_gene_feature, reg_gene, proj_dim, proj_dim/4)
    
    projected_disease_feature = Dense(proj_dim,trainable=True, 
                kernel_regularizer=regularizers.l2(reg_disease),name='disease_projection',
                activation='relu', kernel_initializer='he_normal')(disease_feature)             
    
    #compact_disease_feature = residual_block(residual_disease_feature,reg_disease, proj_dim, proj_dim/4)
    score = dot([projected_gene_feature,projected_disease_feature], 1, name='inner_product')
    # score = dot([projected_gene_feature, projected_disease_feature], 1, name='inner_product')
    # calculate score
    
    # prediction = Dense(1, activation='sigmoid', init='he_uniform', name='prediction')(score)


    model = Model(inputs=[gene_input, disease_input], outputs=score)
    return model

def get_deepimc_model(num_genes, num_diseases, proj_dim, reg_gene, reg_disease):
    # Input variables
    gene_input = Input(shape=(1,), dtype='int32', name='gene_input')
    disease_input = Input(shape=(1,), dtype='int32', name='disease_input')

    # using gene feature matrix for initialization
    gene_feas = h5py.File('gene_features.mat', 'r')
    humannet_features = gene_feas['features'][:,:].T
    gene_feature_size = humannet_features.shape
    assert gene_feature_size[0] + 1 == num_genes

    humannet_features = np.insert(humannet_features, 0, [0]*gene_feature_size[1], 0)
    #humannet_embedding_layer = Embedding(input_dim=gene_feature_size[0]+1, output_dim=gene_feature_size[1], trainable=False)
    #humannet_embedding_layer.build((None,))
    #humannet_embedding_layer.set_weights([humannet_features])
    humannet_embedding_layer = Embedding(input_dim=gene_feature_size[0]+1, output_dim=gene_feature_size[1], trainable=False, weights=[humannet_features])
    # using disease feature matrix for initialization
    feas = h5py.File('disease_features.mat','r')
    omim_features = feas['col_features'][:,:].T
    disease_feature_size = omim_features.shape
    assert disease_feature_size[0] + 1 == num_diseases
    
    omim_features = np.insert(omim_features, 0, [0]*disease_feature_size[1], 0)
    #omim_embedding_layer = Embedding(input_dim=disease_feature_size[0]+1, output_dim=disease_feature_size[1],trainable=False)
    #omim_embedding_layer.build((None,))
    #omim_embedding_layer.set_weights([omim_features])
    omim_embedding_layer = Embedding(input_dim=disease_feature_size[0]+1, output_dim=disease_feature_size[1],trainable=False,weights=[omim_features])
    # get specific features
    gene_feature = Flatten()(humannet_embedding_layer(gene_input))
    disease_feature = Flatten()(omim_embedding_layer(disease_input))
    # disease_feature = K.transpose(disease_feature)
    # projection matrix, using W * H' as initialization
    

    # projection of gene feature and disease feature
    gene_latent_factors = Embedding(input_dim=gene_feature_size[0]+1,
            output_dim=500, trainable=True, embeddings_initializer='he_normal', embeddings_regularizer=regularizers.l2(1e-2))

    
    disease_latent_factors = Embedding(input_dim=disease_feature_size[0]+1,
            output_dim=500, trainable=True, embeddings_initializer='he_normal', embeddings_regularizer=regularizers.l2(1e-2))
    
    # get specific latent factor
    gene_factor = Flatten()(gene_latent_factors(gene_input))
    disease_factor = Flatten()(disease_latent_factors(disease_input))

    projected_gene_feature = Dense(proj_dim,trainable=True, 
                kernel_regularizer=regularizers.l2(reg_gene), name='gene_projection',
                activation='relu', kernel_initializer='he_normal')(gene_feature)
#    residual_gene_feature = residual_block(projected_gene_feature, reg_gene, proj_dim, proj_dim/2) 

    #compact_gene_feature = residual_block(residual_gene_feature, reg_gene, proj_dim, proj_dim/4)
    
    projected_disease_feature = Dense(proj_dim,trainable=True, 
                kernel_regularizer=regularizers.l2(reg_disease),name='disease_projection',
                activation='relu', kernel_initializer='he_normal')(disease_feature)             
    
 #   residual_disease_feature = residual_block(projected_disease_feature, reg_disease, proj_dim, proj_dim/2)
    #compact_disease_feature = residual_block(residual_disease_feature,reg_disease, proj_dim, proj_dim/4)
#    score = dot([residual_gene_feature,residual_disease_feature], 1, name='inner_product')
    left_score = dot([projected_gene_feature, projected_disease_feature], 1, name='IMC_inner_product')
    right_score = dot([gene_factor, disease_factor], 1, name='MF_inner_product')
    score = Add()([left_score,right_score])
    # calculate score
    
    # prediction = Dense(1, activation='sigmoid', init='he_uniform', name='prediction')(score)


    model = Model(inputs=[gene_input, disease_input], outputs=score)
    return model


def get_model(num_genes, num_diseases):
    # Input variables
    gene_input = Input(shape=(1,), dtype='int32', name='gene_input')
    disease_input = Input(shape=(1,), dtype='int32', name='disease_input')

    # using gene feature matrix for initialization
    gene_feas = h5py.File('gene_features.mat', 'r')
    humannet_features = gene_feas['features'][:,:].T

def get_model(num_genes, num_diseases):
    # Input variables
    gene_input = Input(shape=(1,), dtype='int32', name='gene_input')
    disease_input = Input(shape=(1,), dtype='int32', name='disease_input')

    # using gene feature matrix for initialization
    gene_feas = h5py.File('gene_features.mat', 'r')
    humannet_features = gene_feas['features'][:,:].T
    gene_feature_size = humannet_features.shape
    assert gene_feature_size[0] + 1 == num_genes

    humannet_features = np.insert(humannet_features, 0, [0]*gene_feature_size[1], 0)
    humannet_embedding_layer = Embedding(input_dim=gene_feature_size[0]+1, output_dim=gene_feature_size[1], trainable=True)
    humannet_embedding_layer.build((None,))
    humannet_embedding_layer.set_weights([humannet_features])
    
    # using disease feature matrix for initializationmerge
    feas = h5py.File('disease_features.mat','r')
    omim_features = feas['col_features'][:,:].T
    disease_feature_size = omim_features.shape
    assert disease_feature_size[0] + 1 == num_diseases
    
    omim_features = np.insert(omim_features, 0, [0]*disease_feature_size[1], 0)
    omim_embedding_layer = Embedding(input_dim=disease_feature_size[0]+1, output_dim=disease_feature_size[1],trainable=True)
    omim_embedding_layer.build((None,))
    omim_embedding_layer.set_weights([omim_features])
    
    # get specific features
    gene_feature = Flatten(name='')(humannet_embedding_layer(gene_input))
    disease_feature = Flatten()(omim_embedding_layer(disease_input))
    # disease_feature = K.transpose(disease_feature)
    # projection matrix, using W * H' as initialization
    
    
    project_disease = Dense(gene_feature_size[1],trainable=True, name='disease_gene_projection', activation='relu', kernel_initializer=init_weights)(disease_feature)
    
    score = dot([gene_feature, project_disease], 1, name='inner_product')
    # calculate score
    # score = merge([gene_feature, project_disease], mode='mul', name='score')
    
    # prediction = Dense(1, activation='sigmoid', init='he_uniform', name='prediction')(score)


    model = Model(inputs=[gene_input, disease_input], outputs=score)

    return model
def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [],[],[]
    
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in xrange(num_negatives):
            j = np.random.randint(num_items)
            while train.has_key((u, j)):
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    df = pd.DataFrame({'user':user_input, 'item':item_input, 
     'label':labels})
    # df = df.sample(frac=1, random_state=501)
    return df['user'], df['item'], df['label']
def eval_NeuCF(model, topK):
    print('start evaluating NeuCF...')
    print('constructing ScoreMatrix...')
    score_matrix = np.zeros((num_users, num_items), dtype='float')
    items = [i for i in range(num_items)]
    users = np.full(len(items), 1, dtype = 'int32')
    for user_ind in range(1, num_users):
        users = np.full(len(items), user_ind, dtype = 'int32')
        score_from_u = model.predict([users, np.array(items)], 
                                batch_size=12332, verbose=0)
        score_matrix[user_ind,:] = np.reshape(np.array(score_from_u), -1)

    # delte row 0 and column 0
    sio.savemat('NeuCF_ScoreMatrix.mat',{'ScoreMatrix':score_matrix[1:,1:]}) 
    print('constructed finished')
    print('starting evaluating cdf and recall for NeuCF...')
    topR = matlab.double([topK])
    metric_cdf = matlab.double([1])
    metric_recall = matlab.double([0])
    # change directory to locate .m
    # engine = matlab.engine.start_matlab()
    # engine.cd('~/program/neural_collaborative_filtering', nargout=0)
    cdf = engine.eval_for_NeuCF(metric_cdf, topR)
    recall = engine.eval_for_NeuCF(metric_recall, topR)
    return cdf, recall

def label_dependent_loss(alpha):
    def label_dependent(y_true, y_pred):
        return alpha * K.sum(y_true * K.square(y_pred - y_true)) + \
                (1-alpha) * K.sum((1 - y_true ) * K.square(y_pred))
    return label_dependent

def squared_loss(alpha):
    def label_dependent(y_true, y_pred):
        return K.sum(y_true * K.square(y_pred - y_true)) + \
                (1-alpha) * K.sum((1 - y_true ) * K.square(y_pred))
    return label_dependent
    
    
def l21_reg(weight_matrix):
    return 5e-4 * K.sum(K.sqrt(K.sum(K.square(weight_matrix), axis=1)))
    


# TODO root_mean_squared lead to nan
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


if __name__ == '__main__':
    
    args = parse_args()
    num_epochs = args.epochs
    batch_size = args.batch_size
    proj_dim = args.proj_dim
    reg_gene = args.reg_gene
    reg_disease = args.reg_disease
    num_negatives = args.num_neg
    topK = args.r
    alpha = args.alpha
    learning_rate = args.lr
    decay = args.decay
    drop = args.drop
    learner = args.learner
    verbose = args.verbose
    model_pretrain = args.pretrain
    training_model = args.model 
    evaluation_threads = 1#mp.cpu_count()
        
    # Loading data
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = dataset.num_users, dataset.num_items
    
    # Build model
    # model = get_model(num_users, num_items)
    #loss_func = squared_loss(alpha-alpha)
    loss_func = label_dependent_loss(alpha=alpha)
    # loss_func = root_mean_squared_error
    
    
    # specify model
    if training_model.lower() == 'deepimc':
        model = get_deepimc_model(num_users, num_items, proj_dim, reg_gene, reg_disease)
    else:
        model = get_nimc_model(num_users, num_items, proj_dim, reg_gene, reg_disease);

    # specify learner 
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(lr=learning_rate), loss=loss_func)
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss=loss_func)
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate,decay=decay), loss=loss_func)
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss=loss_func)
    # load pretrain model
    if model_pretrain != '':
        model.load_weights(model_pretrain,by_name=True)
    # neumf_pretrain = 'Pretrain/gene_NeuMF_8_[64]_0.050.h5'
    # model.load_weights(neumf_pretrain,by_name=True)

    # print("Load pretrained NeuMF (%s) models done. " %(neumf_pretrain))

    # Load pretrain model
        
    # Init performance
    engine = matlab.engine.start_matlab()
    cur_path = os.getcwd()
    engine.cd(cur_path, nargout=0)
    cdf, recall = eval_NeuCF(model, topK)
    cdf_t, recall_t = cdf[0][topK-1], recall[0][topK-1]
    print('Init: cdf = %.6f, recall = %.6f' % ( cdf_t, recall_t,))
    
    best_cdf, best_recall, best_iter = cdf_t,recall_t,-1

    # Training model
    model.summary()
    for epoch in xrange(1,num_epochs+1):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train, num_negatives)
        
        # Training
        hist = model.fit([np.array(user_input), np.array(item_input)], #input
                         np.array(labels), # labels 
                         batch_size=batch_size, epochs=2, verbose=1, shuffle=True)

        
        t2 = time()
       
        # Evaluation
        if epoch % verbose == 0:
            cdf, recall = eval_NeuCF(model, topK)
            cdf_t, recall_t, loss = cdf[0][topK-1], recall[0][topK-1], hist.history['loss'][0]
            print('Iteration %d [%.1f s]: cdf = %.6f, recall = %.6f, loss = %.8f [%.1f s]' 
                  % (epoch,  t2-t1, cdf_t, recall_t, loss, time()-t2))
            if cdf_t > best_cdf:
                best_cdf, best_recall, best_iter = cdf[0][topK-1], recall[0][topK-1],epoch



    engine.quit()

    # cal ScoreMatrix and save
    score_matrix = np.zeros((num_users, num_items), dtype='float')
    items = [i for i in range(num_items)]
    for user_ind in range(1, num_users):
        users = np.full(len(items), user_ind, dtype = 'int32')
        score_from_u = model.predict([users, np.array(items)], 
                                batch_size=100, verbose=0)
        score_matrix[user_ind,:] = np.reshape(np.array(score_from_u), -1)

    # delte row 0 and column 0
    score_matrix = np.delete(score_matrix,[0], axis=0)
    score_matrix = np.delete(score_matrix,[0], axis=1)
    sio.savemat('NeuCF_ScoreMatrix.mat',{'ScoreMatrix':score_matrix}) 
    
    print("End. Best Iteration %d:  CDF = %.4f, Recall = %.4f. " %(best_iter, best_cdf, best_recall))      


    if args.out > 0:
        model_out_file = 'Pretrain/%s-NeuMC-cdf%.4f-dim%d-alpha%.4f-batch%d.h5' %(args.dataset, best_cdf, proj_dim, alpha, batch_size)
        model.save_weights(model_out_file, overwrite=True)
        print("The best NeuMC model is saved to %s" %(model_out_file))
