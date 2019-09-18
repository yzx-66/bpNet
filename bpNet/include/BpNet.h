#ifndef BPNET_H
#define BPNET_H

#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <ctime>

#define INNODE 2 //输入结点数
#define HIDENODE 4 //隐含结点数
#define OUTNODE 1 //输出结点数
#define LEARNINGRATE 0.9//学习速率（注意：越高虽然越快 也容易误差较大）

/*
* 输入层节点
*/
typedef struct inputNode{
    double value; //输入值
    std::vector<double> weight //输入层单个节点对下一层每个节点的加权值
    , wDeltaSum; //单个加权的不同样本和
}InputNode;

/*
* 输出层节点
*/
typedef struct outputNode{
    double o_value //节点最终值 经过偏移与激活函数后的值
    , rightout //正确输出值
    , bias //偏移量 每个节点只有一个
    , bDeltaSum; //反向传播时 经过计算后的偏移量需要改变的值 因为有多个样本所以是sum
}OutputNode;

/*
*隐含层节点
*/
typedef struct hiddenNode{
    double o_value //节点最终值 经过偏移与激活函数后的值
    , bias //偏移量 每个节点只有一个
    , bDeltaSum; //反向传播时 经过计算后的偏移量需要改变的值 因为有多个样本所以是sum
    std::vector<double> weight //隐藏层单个节点对下一层每个节点的加权值
    , wDeltaSum; //单个加权的不同样本和
}HiddenNode;

/*
* 单个样本
*/
typedef struct sample{
    std::vector<double> in //输入层value的迭代器 里面的数据有输入层节点数个（输入层每个节点的value值 代表一份样本数据中 一个输入属性的值）
    , out; //输出层rightout的迭代器 里面的数据也有输出层层节点数个（输出层每个节点的rightout值 代表一份样本数据 应该输出属性的正确值）
}Sample;

/*
* BP神经网络
*/
class BpNet{
    public:
        BpNet(); //构造函数 用来初始化加权和偏移
        void fp(); //单个样本前向传播
        void bp(); //单个样本后向传播
        void doTraining(std::vector<Sample> sampleGroup, double threshold,int mostTimes);//训练（更新 weight, bias）
        void afterTrainTest(std::vector<Sample>& testGroup); //神经网络学习后进行预测
        void setInValue (std::vector<double> inValue);     //设置学习样本输入
        void setOutRightValue(std::vector<double> outRightValue);    //设置学习样本输出

    public://设置成public就不用get、set麻烦
        double error; //误差率
        InputNode* inputLayer[INNODE]; //输入层（任何模型都只有一层）
        OutputNode* outputLayer[OUTNODE]; //输出层（任何模型都只有一层）
        HiddenNode* hiddenLayer[HIDENODE]; // 隐含层（如果有多层是二维数组 这里只有一个隐藏层所以一维数组）
};


#endif // BPNET_H
