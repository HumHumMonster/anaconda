import numpy as np
from scipy.optimize import minimize

# 预处理
def prepare_for_training(data, polynomial_degree = 0 , sinusoid_degree = 0 , normalize_data = True) :
    num_examples = data.shape[0] ;
    
    data_processed = np.copy(data) ;

    # 预处理
    features_mean = 0 ;
    features_deviation = 0 ;
    if normalize_data :
        (
            data_normalized ,
            features_mean ,
            features_deviation ,
        ) = normalize(data_processed) 
    data_processed = data_normalized

    # 加上一列1
    data_processed = np.hstack((np.ones((num_examples, 1)), data_processed)) ;

    return data_processed, features_mean, features_deviation ;

def normalize(features) :
    features_normalized = np.copy(features).astype(float) ;
    # 计算均值
    features_mean = np.mean(features, 0) ;
    # 计算标准差
    features_deviation = np.std(features, 0) ;

    if features.shape[0] > 1 :
        features_normalized -= features_mean ;

    # 防止除以0
    features_deviation[features_deviation == 0] = 1 ;
    features_normalized /= features_deviation

    return features_normalized, features_mean, features_deviation ;

def sigmoid (martix) :
    return 1 / (1 + np.exp(-martix)) ;

class LogisticRegression :
    def __init__(self, data, labels, polynomial_degree = 0, sinusoid_degree = 0, normalize_data = True) -> None:

        (data_processed,
         features_mean,
         features_deviation ) = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data) ;

        self.data = data_processed ;
        self.labels = labels ;
        self.features_mean = features_mean ;
        self.features_deviation = features_deviation ;
        self.polynomial_degree = polynomial_degree ;
        self.sinusoid_degree = sinusoid_degree ;
        self.normalize_data = normalize_data ;

        
        num_features = self.data.shape[1] ;
        num_unique_labels = np.unique(labels).shape[0] ;
        self.theta = np.zeros((num_unique_labels, num_features)) ;

    def train (self, max_iterations=1000) :
        cost_histories = [] ;
        num_features = self.data.shape[1] ;
        for label_index, uinque_label in enumerate(self.unique_labels) :
            current_initial_theta = np.copy(self.theta[label_index].reshape(num_features, 1)) ;
            current_labels = (self.labels == unique_label).astype(float) ;
            (current_theta, cost_histories) = LogisticRegression.gradient_descent(self.data, current_labels, current_initial_theta, max_iterations) ;

    @staticmethod
    def gradient_descent(data, labels, current_initial_theta, max_iterations) :
        cost_history = [] ;
        num_features = data.shape[1] ;
        minimize(
            # 要优化的目标
            lambda  current_theta: LogisticRegression.cost_function(data, labels, current_initial_theta.reshape(num_features, 1))

        )

    @staticmethod
    def cost_function(data, labels, theta) :
        num_examples = data.shape[0] ;
        predictions = LogisticRegression.hypothesis(data, theta) ;
        y_is_set_cost = np.dot(labels[labels == 1].T , np.log(predictions[labels == 1])) ;
        y_is_not_set_cost = np.dot(1 - labels[labels == 0].T , np.log(1 - predictions[labels == 0])) ;
        cost = (-1 / num_examples) * (y_is_set_cost + y_is_not_set_cost) ;
        return cost ;
    

    @staticmethod
    def hypothsis(data, theta) :
        predictions = sigmoid(np.dot(data, theta)) ;

        return predictions ;
        

