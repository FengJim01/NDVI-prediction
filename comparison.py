import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    #data
    train1=np.load("./model/mix/history.npy")
    test1=np.load("./model/mix/test_loss.npy")
    train2=np.load("./model/16/history.npy")
    test2=np.load("./model/16/test_loss.npy")
    train3=np.load("./model/17/history.npy")
    test3=np.load("./model/17/test_loss.npy")

    fig = plt.figure(figsize=(20,10))
    # fig.suptitle("Loss Function Comparison",fontsize=20)
    ax = plt.axes()
    # train1
    train1_x = np.linspace(1, train1.size, num=train1.size)
    ax.plot(train1_x, np.squeeze(train1), 'r',lw=5, label="CNN-1 訓練資料損失函式")
    ax.scatter(train1_x, np.squeeze(train1), c='r', s=70)
    # test1
    test1_x = np.linspace(1, test1.size, num=test1.size)
    ax.plot(test1_x, test1, 'r', lw=3,linestyle="--", label='CNN-1 測試資料損失函式')
    ax.scatter(test1_x, test1, c='r', s=70)

    # train2
    train2_x = np.linspace(1, train2.size, num=train2.size)
    ax.plot(train2_x, np.squeeze(train2), 'b',lw=3, label="CNN-2 訓練資料損失函式")
    ax.scatter(train2_x, np.squeeze(train2), c='b', s=70)
    # test2
    test2_x = np.linspace(1, test2.size, num=test2.size)
    ax.plot(test2_x, test2, 'b',lw=3, linestyle="--", label='CNN-2 測試資料損失函式')
    ax.scatter(test2_x, test2, c='b', s=70)

    # train3
    train3_x = np.linspace(1, train3.size, num=train3.size)
    ax.plot(train3_x, np.squeeze(train3), 'g',lw=3, label="CNN-3 訓練資料損失函式")
    ax.scatter(train3_x, np.squeeze(train3), c='g', s=70)

    # test3
    test3_x = np.linspace(1, test3.size, num=test3.size)
    ax.plot(test3_x, test3, 'g',lw=3, linestyle="--", label='CNN-3 測試資料損失函式')
    ax.scatter(test3_x, test3, c='g', s=70)

    # label
    # ax.legend(loc='center',fontsize=40)
    plt.xticks(np.arange(0,21,2),fontsize=40)
    plt.yticks(fontsize=40)
    # plt.xlabel("迭代",fontsize=60)
    # plt.ylabel("MSE",fontsize=60)
    plt.rcParams['font.sans-serif'] = ["DFKai-SB"]
    plt.rcParams['axes.unicode_minus'] = False
    fig.savefig("./comparison.png")
    # plt.show()