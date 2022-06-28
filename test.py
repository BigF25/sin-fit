import numpy as np
import

class ApproachNetwork:
    def __init__(self, hidden_size=100, output_size=1):
    self.params = {'W1': np.random.random((1, hidden_size)),
                   'B1': np.zeros(hidden_size),
                   'W2': np.random.random((hidden_size, output_size)),
                   'B2': np.zeros(output_size)}
    @staticmethod
    def generate_data(fun, is_noise=True, axis=np.array([-1, 1, 100])):
        """
        产生数据集
        :param fun: 这个是你希望逼近的函数功能定义，在外面定义一个函数功能方法，把功能方法名传入即可 
        :param is_noise: 是否需要加上噪点，True是加，False表示不加
        :param axis: 这个是产生数据的起点，终点，以及产生多少个数据
        :return: 返回数据的x, y
        """
        np.random.seed(0)
        x = np.linspace(axis[0], axis[1], axis[2])[:, np.newaxis]
        x_size = x.size
        y = np.zeros((x_size, 1))
        if is_noise:
            noise = np.random.normal(0, 0.1, x_size)
        else:
            noise = None

        for i in range(x_size):
            if is_noise:
                y[i] = fun(x[i]) + noise[i]
            else:
                y[i] = fun(x[i])

        return x, y
    # 逼近函数 f(x)=sin(x)
    def fun_sin(x0):
        return math.sin(x0)

    x, y = network.generate_data(fun_sin, False, axis=np.array([-3, 3, 100]))
    ax = plt.gca()
    ax.set_title('data points')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.scatter(x, y)
    plt.show()



