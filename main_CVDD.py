import click
import optuna
import torch
import logging
import random
import numpy as np
from sklearn.model_selection import train_test_split

from cvdd import CVDD
from datasets.CSIC2010 import CSIC2010
from utils.config import Config
from utils.visualization import plot_matrix_heatmap, plot_joyplot
from utils.misc import print_text_samples, print_top_words, get_correlation_matrix
#from cvdd import CVDD

################################################################################
# Settings
################################################################################
from datasets.preprocessing import GetDataSet


@click.command()
@click.argument('dataset_name', type=click.Choice(['csic2010','reuters', 'newsgroups20', 'imdb']))
@click.argument('net_name', type=click.Choice(['cvdd_Net']))
@click.argument('xp_path', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--load_config', type=click.Path(exists=True), default=None,
              help='Config JSON-file path (default: None).')
@click.option('--load_model', type=click.Path(exists=True), default=None,
              help='Model file path (default: None).')
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
@click.option('--tokenizer', default='spacy', type=click.Choice(['spacy', 'bert']), help='Select text tokenizer.')
@click.option('--clean_txt', is_flag=True, help='Specify if text should be cleaned in a pre-processing step.')
@click.option('--embedding_size', type=int, default=None, help='Size of the word vector embedding.')
@click.option('--pretrained_model', default=None,
              type=click.Choice([None, 'GloVe_6B', 'GloVe_42B', 'GloVe_840B', 'GloVe_twitter.27B', 'FastText_en',
                                 'bert']),
              help='Load pre-trained word vectors or language models to initialize the word embeddings.')
@click.option('--ad_score', default='context_dist_mean', type=click.Choice(['context_dist_mean', 'context_best']),
              help='Choose the AD score function')
@click.option('--n_attention_heads', type=int, default=1, help='Number of attention heads in self-attention module.')
@click.option('--attention_size', type=int, default=100, help='Self-attention module dimensionality.')
@click.option('--lambda_p', type=float, default=1.0,
              help='Hyperparameter for context vector orthogonality regularization P = (CCT - I)')
@click.option('--alpha_scheduler', default='logarithmic', type=click.Choice(['soft', 'linear', 'logarithmic', 'hard']),
              help='Set annealing strategy for temperature hyperparameter alpha.')
@click.option('--optimizer_name', type=click.Choice(['adam']), default='adam',
              help='Name of the optimizer to use for training.')
@click.option('--lr', type=float, default=0.001,
              help='Initial learning rate for training. Default=0.001')
@click.option('--n_epochs', type=int, default=50, help='Number of epochs to train.')
@click.option('--lr_milestone', type=int, default=0, multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--batch_size', type=int, default=64, help='Batch size for mini-batch training.')
@click.option('--weight_decay', type=float, default=0.5e-6,
              help='Weight decay (L2 penalty) hyperparameter.')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--n_threads', type=int, default=0,
              help='Sets the number of OpenMP threads used for parallelizing CPU operations')
@click.option('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
def main(dataset_name, net_name, xp_path, data_path, load_config, load_model, device, seed, tokenizer, clean_txt,
         embedding_size, pretrained_model, ad_score, n_attention_heads, attention_size, lambda_p, alpha_scheduler,
         optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay, n_jobs_dataloader, n_threads,
         normal_class):
    """
    Context Vector Data Description (CVDD): An unsupervised anomaly detection method for text.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """

    # Get configuration
    cfg = Config(locals().copy())

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print paths
    logger.info('Log file is %s.' % log_file)
    logger.info('Data path is %s.' % data_path)
    logger.info('Export path is %s.' % xp_path)

    # Print experimental setup
    logger.info('Dataset: %s' % dataset_name)
    logger.info('Normal class: %d' % normal_class)# 哪个是正常数据
    logger.info('Network: %s' % net_name)         #使用的网络  cvdd_Net是作者写的网络
    logger.info('Tokenizer: %s' % cfg.settings['tokenizer'])  #spacy 来做token
    logger.info('Clean text in pre-processing: %s' % cfg.settings['clean_txt'])   #预处理中清理文本？
    if cfg.settings['embedding_size'] is not None:                                #词嵌入维度
        logger.info('Word vector embedding size: %d' % cfg.settings['embedding_size'])
    logger.info('Load pre-trained model: %s' % cfg.settings['pretrained_model'])     #预训练模型 GloVe_6B

    # Print CVDD configuration)
    logger.info('Anomaly Score: %s' % cfg.settings['ad_score'])    #context_dist_mean  上下文特征向量的距离作为异常值评估？
    logger.info('Number of attention heads: %d' % cfg.settings['n_attention_heads'])   #多头attention 有几个头
    logger.info('Attention size: %d' % cfg.settings['attention_size'])                 #attention size 150
    logger.info('Orthogonality regularization hyperparameter: %.3f' % cfg.settings['lambda_p'])  #Orthogonality regularization hyperparameter正交规范化参数？
    logger.info('Temperature alpha annealing strategy: %s' % cfg.settings['alpha_scheduler'])    #模拟退火算法  对数？

    # If specified, load experiment config from JSON-file
    if load_config: #None
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)

    # Set seed for reproducibility
    if cfg.settings['seed'] != -1:  #有seed可复现
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        torch.cuda.manual_seed(cfg.settings['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        logger.info('Set seed to %d.' % cfg.settings['seed'])

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():   #CUDA版本
        device = 'cpu'
    logger.info('Computation device: %s' % device)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)
    if n_threads > 0: #不用多线程
        torch.set_num_threads(n_threads)
        logger.info('Number of threads used for parallelizing CPU operations: %d' % n_threads)

    # Load data
    normal_data,text2tokensdic = GetDataSet(data_path+"/normal.txt",58,device=device)

    anomalous_data ,_ = GetDataSet(data_path + "/anomalous.txt", 65, device=device , text2tokensdic=text2tokensdic)
    tmp = torch.load("./123.pt")
    xxx=tmp == text2tokensdic      #测试数据集每次训练是不是一样的
    print("与上一次Token比是否一样",xxx)
    torch.save(text2tokensdic, "./123.pt")

    tmp1 = torch.load("./1234.pt")
    xxx1 = tmp1.equal(normal_data.token_text)   # 测试数据集每次训练是不是一样的
    print("与上一次Tokened比是否一样",xxx1)
    torch.save(normal_data.token_text, "./1234.pt")

    train_normal, test_normal, _, _ = train_test_split(normal_data.token_text, normal_data.token_text, test_size=0.3, random_state=seed)

    train_normal_data = CSIC2010(train_normal,device=device)
    test_normal_data = CSIC2010(test_normal,device=device)


    # dataset = load_dataset(dataset_name, data_path, normal_class, cfg.settings['tokenizer'],
    #                        clean_txt=cfg.settings['clean_txt'])
    dataset=None
    # Initialize CVDD model and set word embedding
    cvdd = CVDD(cfg.settings['ad_score'])#做了一个空壳准备了几个参数向量
    cvdd.set_network(net_name=net_name,
                     dataset=train_normal_data,
                     vocab_size=len(text2tokensdic),
                     #pretrained_model=cfg.settings['pretrained_model'],
                     embedding_size=cfg.settings['embedding_size'],
                     attention_size=cfg.settings['attention_size'],
                     n_attention_heads=cfg.settings['n_attention_heads'])

    # If specified, load model parameters from already trained model
    if load_model:
        cvdd.load_model(import_path=load_model, device=device)
        logger.info('Loading model from %s.' % load_model)

    # Train model on dataset
    cvdd.train(train_normal_data,
               optimizer_name=cfg.settings['optimizer_name'],
               lr=cfg.settings['lr'],
               n_epochs=cfg.settings['n_epochs'],
               lr_milestones=cfg.settings['lr_milestone'],#epoch到达这个值时对lr调整？
               batch_size=cfg.settings['batch_size'],
               lambda_p=cfg.settings['lambda_p'],
               alpha_scheduler=cfg.settings['alpha_scheduler'],
               weight_decay=cfg.settings['weight_decay'],
               device=device,
               n_jobs_dataloader=n_jobs_dataloader)

    # Test model
    cvdd.test(test_normal_data,anomalous_data, device=device, n_jobs_dataloader=n_jobs_dataloader)
    """
    # Print most anomalous and most normal test samples
    indices, labels, scores, heads = zip(*cvdd.results['test_scores'])
    indices, scores = np.array(indices), np.array(scores)
    sort_idx = np.argsort(scores).tolist()  # sorted from lowest to highest anomaly score
    idx_sorted = indices[sort_idx]
    idx_normal = idx_sorted[:50].tolist()
    idx_outlier = idx_sorted[-50:].tolist()[::-1]
    att_weights = cvdd.test_att_weights
    att_weights_sorted = [att_weights[i] for i in sort_idx]
    att_weights_normal = att_weights_sorted[:50]
    att_weights_outlier = att_weights_sorted[-50:][::-1]
    heads_sorted = [heads[i] for i in sort_idx]
    heads_normal = heads_sorted[:50]
    heads_outlier = heads_sorted[-50:][::-1]

    print_text_samples(dataset.test_set, dataset.encoder, idx_normal, export_file=xp_path + '/normals',
                       att_heads=heads_normal, weights=att_weights_normal, title='Most normal examples')
    print_text_samples(dataset.test_set, dataset.encoder, idx_outlier, export_file=xp_path + '/outliers',
                       att_heads=heads_outlier, weights=att_weights_outlier, title='Most anomalous examples')

    # Print top words per context
    train_top_words, test_top_words = cvdd.train_top_words, cvdd.test_top_words
    print_top_words(train_top_words, export_file=xp_path + '/top_words_train',
                    title='Top words per context in train set')
    print_top_words(test_top_words, export_file=xp_path + '/top_words_test',
                    title='Top words per context in test set')

    # Print context vector correlation matrix
    if cfg.settings['n_attention_heads'] > 1:
        context_vectors = np.array(cvdd.results['context_vectors'])
        corr_mat = get_correlation_matrix(context_vectors)
        plot_matrix_heatmap(corr_mat, title='Context vectors correlation matrix',
                            export_pdf=xp_path + '/context_vecs_matrix')

    # Print attention matrix heatmaps
    if cfg.settings['n_attention_heads'] > 1:
        train_att_matrix = cvdd.results['train_att_matrix']
        test_att_matrix = cvdd.results['test_att_matrix']
        train_att_matrix, test_att_matrix = np.array(train_att_matrix), np.array(test_att_matrix)
        plot_matrix_heatmap(train_att_matrix, title='Self-attention heads correlation matrix',
                            export_pdf=xp_path + '/att_heatmap_train')
        plot_matrix_heatmap(test_att_matrix, title='Self-attention heads correlation matrix',
                            export_pdf=xp_path + '/att_heatmap_test')

    # Plot distributions of distances to context vector per attention head
    train_dists, test_dists = cvdd.train_dists, cvdd.test_dists
    plot_joyplot(train_dists, title='Distances from context vector per attention head',
                 export_pdf=xp_path + '/dists_train')
    plot_joyplot(test_dists[np.array(labels) == 0, :], title='Distances from context vector per attention head',
                 export_pdf=xp_path + '/dists_test_normals')
    if np.sum(np.array(labels)) > 0:
        plot_joyplot(test_dists[np.array(labels) == 1, :], title='Distances from context vector per attention head',
                     export_pdf=xp_path + '/dists_test_outliers')

    # Save results, model, and configuration
    cvdd.save_results(export_json=xp_path + '/results.json')
    cvdd.save_model(export_path=xp_path + '/model.tar')
    cfg.save_config(export_json=xp_path + '/config.json')
    """

    print(cvdd.trainer.test_auc)
    return cvdd.trainer.test_auc


# csic2010 cvdd_Net ./log/test_csic2010 ./data --device cpu --seed 1 --clean_txt --embedding_size 300 --pretrained_model GloVe_6B
# --ad_score context_dist_mean --n_attention_heads 3 --attention_size 150 --lambda_p 1.0 --alpha_scheduler logarithmic --n_epochs 100
# --lr 0.01 --lr_milestone 40  --normal_class 6
def objective(trial):
    rf_params = {
        "dataset_name": "csic2010",
         "net_name" :"cvdd_Net",
         "xp_path": "./log/test_csic2010",
         "data_path" : "./data",
         "load_config" : None,
         "load_model" :None,
         "device" : "cpu",
         "seed" : 1,
         "tokenizer" : "spacy",
         "clean_txt" : "True",
         "embedding_size" : trial.suggest_int(name="embedding_size", low=100, high=300),
         "pretrained_model" : "",
         "ad_score" :"context_dist_mean",
         "n_attention_heads" : trial.suggest_int(name="n_attention_heads", low=3, high=100),
         "attention_size":trial.suggest_int(name="attention_size", low=50, high=300),
         "lambda_p": 1.0,
         "alpha_scheduler" : "logarithmic",
         "optimizer_name" : "adam",
         "lr" : 0.01,
         "n_epochs" :trial.suggest_int(name="n_epochs", low=5, high=100),
         "lr_milestone" : 40,
         "batch_size":trial.suggest_categorical("batch_size", choices=[32,64]),
         "weight_decay":0.5e-6,
         "n_jobs_dataloader":0,
         "n_threads": 0,
         "normal_class" : 6,

    }

    AUC = main(**rf_params)
    return AUC
    """
    Context Vector Data Description (CVDD): An unsupervised anomaly detection method for text.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """


# csic2010 cvdd_Net ./log/test_csic2010 ./data --device cpu --seed 1 --clean_txt --embedding_size 300 --pretrained_model GloVe_6B
# --ad_score context_dist_mean --n_attention_heads 3 --attention_size 150 --lambda_p 1.0 --alpha_scheduler logarithmic --n_epochs 100
# --lr 0.01 --lr_milestone 40  --normal_class 6
# if __name__ == '__main__':
#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective, n_trials=10000)
#

if __name__ == '__main__':
  main()

