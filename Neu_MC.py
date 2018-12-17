import tensorflow as tf
import keras
from keras import backend as K
import numpy as np
import h5py
from keras.models import Model
from keras.layers import Embedding, Input, Dense, Activation, dot
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Dropout,BatchNormalization
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
import argparse
import scipy.io as sio
from Dataset import Dataset
import matlab.engine
from time import time
from evaluate import evaluate_model
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='gene',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size of MF model.')
    parser.add_argument('--layers', nargs='?', default='[64]',
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_mf', type=float, default=0.01,
                        help='Regularization for MF embeddings.')                    
    parser.add_argument('--reg_layers', nargs='?', default='[0.01]',
                        help="Regularization for each MLP layer. reg_layers[0] is the regularization for embeddings.")
    parser.add_argument('--num_neg', type=int, default=5,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--mf_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MF part. If empty, no pretrain will be used')
    parser.add_argument('--mlp_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
    return parser.parse_args()

def init_weights(shape, name=None):
    feas = h5py.File('projection.mat','r')
    projection = feas['proj'][:,:]
    return K.variable(projection)


def get_nn_model(num_genes, num_diseases, mc_dim):
    # Input variables
    gene_input = Input(shape=(1,), dtype='int32', name='gene_input')
    disease_input = Input(shape=(1,), dtype='int32', name='disease_input')

    # using gene feature matrix for initialization
    gene_feas = h5py.File('gene_features.mat', 'r')
    humannet_features = gene_feas['features'][:,:].T
    gene_feature_size = humannet_features.shape
    assert gene_feature_size[0] + 1 == num_genes

    humannet_features = np.insert(humannet_features, 0, [0]*gene_feature_size[1], 0)
    humannet_embedding_layer = Embedding(input_dim=gene_feature_size[0]+1, output_dim=gene_feature_size[1], trainable=False)
    humannet_embedding_layer.build((None,))
    humannet_embedding_layer.set_weights([humannet_features])
    
    # using disease feature matrix for initializationmerge
    feas = h5py.File('disease_features.mat','r')
    omim_features = feas['col_features'][:,:].T
    disease_feature_size = omim_features.shape
    assert disease_feature_size[0] + 1 == num_diseases
    
    omim_features = np.insert(omim_features, 0, [0]*disease_feature_size[1], 0)
    omim_embedding_layer = Embedding(input_dim=disease_feature_size[0]+1, output_dim=disease_feature_size[1],trainable=False)
    omim_embedding_layer.build((None,))
    omim_embedding_layer.set_weights([omim_features])
    
    # get specific features
    gene_feature = Flatten(name='')(humannet_embedding_layer(gene_input))
    disease_feature = Flatten()(omim_embedding_layer(disease_input))
    # disease_feature = K.transpose(disease_feature)
    # projection matrix, using W * H' as initialization
    
    # projection of gene feature and disease feature
    
    projected_gene_feature = Dense(mc_dim,trainable=True, name='gene_projection', activation='relu', kernel_initializer='he_init')(gene_feature)
    
    projected_disease_feature = Dense(mc_dim,trainable=True, name='disease_projection', activation='relu', kernel_initializer='he_init')(disease_feature)
    score = dot([projected_gene_feature, projected_disease_feature], False, name='inner_product')
    # calculate score
    # score = merge([gene_feature, project_disease], mode='mul', name='score')
    
    # prediction = Dense(1, activation='sigmoid', init='he_uniform', name='prediction')(score)


    model = Model(inputs=[gene_input, disease_input], outputs=score)
    


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
    
    score = dot([gene_feature, project_disease], False, name='inner_product')
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
    df = df.sample(frac=1, random_state=501)
    return df['user'], df['item'], df['label']
def eval_NeuCF(model):
    print('start evaluating NeuCF...')
    print('constructing ScoreMatrix...')
    score_matrix = np.zeros((num_users, num_items), dtype='float')
    items = [i for i in range(num_items)]
    users = np.full(len(items), 1, dtype = 'int32')
    dot_out_model = Model(inputs=model.input, \
                outputs=model.get_layer('inner_product').output)
    dot  = dot_out_model.predict([users, np.array(items)], batch_size=12332,verbose=0)
    for user_ind in range(1, num_users):
        users = np.full(len(items), user_ind, dtype = 'int32')
        score_from_u = model.predict([users, np.array(items)], 
                                batch_size=12332, verbose=0)
        score_matrix[user_ind,:] = np.reshape(np.array(score_from_u), -1)

    # delte row 0 and column 0
    sio.savemat('NeuCF_ScoreMatrix.mat',{'ScoreMatrix':score_matrix[1:,1:]}) 
    print('constructed finished')
    print('starting evaluating cdf and recall for NeuCF...')
    topR = matlab.double([100])
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

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


if __name__ == '__main__':
    
    args = parse_args()
    num_epochs = args.epochs
    batch_size = args.batch_size
    mf_dim = args.num_factors
    layers = eval(args.layers)
    reg_mf = args.reg_mf
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learning_rate = args.lr
    learner = args.learner
    verbose = args.verbose
    mf_pretrain = args.mf_pretrain
    mlp_pretrain = args.mlp_pretrain

    topK = 100
    evaluation_threads = 1#mp.cpu_count()
    model_out_file = 'Pretrain/NeuMC.h5'
        
    # Loading data
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = dataset.num_users, dataset.num_items
    
    # Build model
    model = get_model(num_users, num_items)
    # loss_func = 'binary_crossentropy'
    loss_func = label_dependent_loss(alpha=1-5e-3)   
    loss_func = root_mean_squared_error
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(lr=learning_rate), loss=loss_func)
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss=loss_func)
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss=loss_func)
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss=loss_func)
    
    # load pretrain model
    
    # neumf_pretrain = 'Pretrain/gene_NeuMF_8_[64]_0.050.h5'
    # model.load_weights(neumf_pretrain,by_name=True)

    # print("Load pretrained NeuMF (%s) models done. " %(neumf_pretrain))

    # Load pretrain model
        
    # Init performance
    engine = matlab.engine.start_matlab()
    engine.cd('~/program/neural_collaborative_filtering', nargout=0)
    cdf, recall = eval_NeuCF(model)
    cdf_t, recall_t = cdf[0][99], recall[0][99]
    print('Init: cdf = %.6f, recall = %.6f' % ( cdf_t, recall_t,))
    
    best_cdf, best_recall, best_iter = cdf_t,recall_t,-1

    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f' %(hr, ndcg))
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    # Training model
    
    model.summary()
    for epoch in xrange(num_epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train, num_negatives)
        
        # Training
        hist = model.fit([np.array(user_input), np.array(item_input)], #input
                         np.array(labels), # labels 
                         batch_size=batch_size, nb_epoch=1, verbose=1, shuffle=True)

        
        t2 = time()
       
        # Evaluation
        if epoch % 4 == 0:
            cdf, recall = eval_NeuCF(model)
            cdf_t, recall_t, loss = cdf[0][99], recall[0][99], hist.history['loss'][0]
            print('Iteration %d [%.1f s]: cdf = %.6f, recall = %.6f, loss = %.8f [%.1f s]' 
                  % (epoch,  t2-t1, cdf_t, recall_t, loss, time()-t2))
            if cdf_t > best_cdf:
                best_cdf, best_recall, best_iter = cdf[0][99], recall[0][99],           epoch
                if args.out > 0:
                    model_out_file = 'Pretrain/%s_NeuMF_%d_%s_%.3f.h5' %(args.dataset, mf_dim, args.layers, best_cdf)
                    model.save_weights(model_out_file, overwrite=True)

        if epoch % 2 == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.6f [%.1f s]' 
                  % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
            if hr > best_hr or ndcg > best_ndcg:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                # if args.out > 0:
                #     model.save_weights(model_out_file, overwrite=True)
    
    # engine.quit()

    # cal ScoreMatrix 
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
    
    # print("End. Best Iteration %d:  cdf = %.4f, recall = %.4f. " %(best_iter, best_cdf, best_recall))      
    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))      
    print("End. Best Iteration %d:  CDF = %.4f, Recall = %.4f. " %(best_iter, best_cdf, best_recall))      

    # print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
    if args.out > 0:
        model_out_file = 'Pretrain/%s_NeuMF_%d_%s_%.3f.h5' %(args.dataset, mf_dim, args.layers, best_cdf)
        print("The best NeuMF model is saved to %s" %(model_out_file))