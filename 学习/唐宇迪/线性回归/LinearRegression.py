import numpy as np

class LinearRegression :

    # 1、对数据进行初始化
    # 2、先得到所有的特征个数
    # 3、初始化参数矩阵
    def __init__(self, data, labels, polynomial_degree = 0, sinusoid_degree = 0, normalize_data = True) -> None:
        # 先对数据做预处理
        # 处理后的数据
        # 特征_均值
        # 特征_偏差
        (data_processed ,
        features_mean ,
        feature_deviation) = None ;

        self.data = data_processed ;
        self.labels = labels ;
        self.features_mean = features_mean ;
        self.feature_deviation = feature_deviation ;
        self.polynomial_degree = polynomial_degree ;
        self.sinusoid_degree = sinusoid_degree ;
        self.normalize_data = normalize_data ;

        num_features = self.data.shape[1] ;
        # theta初始化为一列 0
        self.theta = np.zeros((num_features , 1)) ;
    




    def train (self, alpha, num_iterations = 500) :
        cost_history = self.gradient_descent(alpha, num_iterations) ;
        return cost_history ;

    # 梯度下降
    def gradient_descent(self, alpha, num_iterations) :
        # 迭代进行 ;
        # 记录每次的损失值
        cost_history = [] ;
        for _ in range (num_iterations) :
            self.gradient_step(alpha) ;
            cost_history.append(self.cost_function(self.data, self.labels)) ;
        return cost_history ;


    # 每一步
    def gradient_step(self, alpha) :
        # 梯度下降参数更新计数方法

        num_example = self.data.shape[0] ;
        prediction = LinearRegression.hypothesis(self.data, self.theta) ;
        delta = prediction - self.labels ;
        theta = self.theta ;
        theta = theta - alpha * (1 / num_example) * (np.dot(delta.T, self.data)).T   ;

    def cost_function(self, data, labels) :
        num_examples = data.shape[0] ;
        delta = LinearRegression.hypothesis(self.data, self.theta) - labels ;
        cost = (1 / 2) * np.dot(delta.T , delta) ;
        return cost[0][0] ;



    # 算预测值
    @staticmethod
    def hypothesis(data, theta) :
        predictions = np.dot(data, theta) ;
        # 返回的预测值是一列
        return predictions ;
            
    def get_cost (self, data, labels) :
        data_processed = prepare_for_training (data ,
                                               )


