#include "BpNet.h"

using namespace std;

/*
* 产生-1~1的随机数
*/
inline double getRandom()   {
    return ((2.0*(double)rand()/RAND_MAX) - 1);
}

/*
* sigmoid 函数（激活函数 要保证单调 且只有一个变量）
*/
inline double sigmoid(double x){
    //一般bp用作分类的话都用该函数
    double ans = 1 / (1+exp(-x));
    return ans;
}


/*
* 初始化（给加权或者偏移赋初值）
*/
BpNet::BpNet(){
    srand((unsigned)time(NULL));
    // error初始值，只要能保证大于阀值进入训练就可以
    error = 100.f;

    /*
     *初始化输入层每个节点对下一层每个节点的加权
     */
    for (int i = 0; i < INNODE; i++){
        inputLayer[i] = new InputNode();
        for (int j = 0; j < HIDENODE; j++){
            inputLayer[i]->weight.push_back(getRandom());
            inputLayer[i]->wDeltaSum.push_back(0.f);
        }
    }

    /*
     *初始化隐藏层每个节点对下一层每个节点的加权
     *初始化隐藏层每个节点的偏移
     */
    for (int i = 0; i < HIDENODE; i++){
        hiddenLayer[i] = new HiddenNode();
        hiddenLayer[i]->bias = getRandom();

        //初始化加权
        for (int j = 0; j < OUTNODE; j++){
            hiddenLayer[i]->weight.push_back(getRandom());
            hiddenLayer[i]->wDeltaSum.push_back(0.f);
        }
    }

    /*
     *初始化输出层每个节点的偏移
     */
    for (int i = 0; i < OUTNODE; i++){
        outputLayer[i] = new OutputNode();
        outputLayer[i]->bias = getRandom();
    }
}


/*
* 正向传播 获取一个样本从输入到输出的结果
*/
void BpNet::fp(){
    /*
     *隐藏层向输入层获取数据
     */
    //遍历隐藏层节点
    for (int i = 0; i < HIDENODE; i++){
        double sum = 0.f;

        //遍历输入层每个节点
        for (int j = 0; j < INNODE; j++){
            sum += inputLayer[j]->value * inputLayer[j]->weight[i];
        }

        //增加偏移
        sum += hiddenLayer[i]->bias;
        //调用激活函数 设置o_value
        hiddenLayer[i]->o_value = sigmoid(sum);
   }

    /*
     *输出层向隐藏层获取数据
     */
    //遍历输出层节点
    for (int i = 0; i < OUTNODE; i++){
        double sum = 0.f;

        //遍历隐藏层节点
        for (int j = 0; j < HIDENODE; j++){
            sum += hiddenLayer[j]->o_value * hiddenLayer[j]->weight[i];
        }

        sum += outputLayer[i]->bias;
        outputLayer[i]->o_value = sigmoid(sum);
    }
}


/*
* 反向传播 从输出层再反向
*
* 该方法目的是返回多个样本 加权应该变化值的和wDeltaSum、偏移应该变化值的和bDeltaSum
* 在训练时根据样本数求平均值 用该平均值修改加权、偏移
*
*/
void BpNet::bp(){
    /*
     *求误差值error
     */
    for (int i = 0; i < OUTNODE; i++){
        double tmpe = fabs(outputLayer[i]->o_value-outputLayer[i]->rightout);
        error += tmpe * tmpe / 2;
    }


    /*
     *求输出层偏移的变化值
     */
    for(int i=0;i<OUTNODE;i++){
        //偏移应该变化的值 参照b2公式
        double bDelta=(-1)*(outputLayer[i]->rightout-outputLayer[i]->o_value)*outputLayer[i]->o_value*(1-outputLayer[i]->o_value);
        outputLayer[i]->bDeltaSum+=bDelta;
    }

    /*
     *求对输出层加权的变化值
     */
    for (int i = 0; i < HIDENODE; i++){
        for(int j=0;j<OUTNODE;j++){
            //加权应该变化的值 参照w9公式
            double wDelta=(-1)*(outputLayer[j]->rightout-outputLayer[j]->o_value)*outputLayer[j]->o_value*(1-outputLayer[j]->o_value)*hiddenLayer[i]->o_value;
            hiddenLayer[i]->wDeltaSum[j]+=wDelta;
        }
    }

    /*
     *求隐藏层偏移
     */
     for(int i=0;i<HIDENODE;i++){
        double sum=0;//因为是遍历输出层节点 不可以确定有多少个输出节点 参照b1公式的第一个公因式
        for(int j=0;j<OUTNODE;j++){
            sum+=(-1)*(outputLayer[j]->rightout-outputLayer[j]->o_value)*outputLayer[j]->o_value*(1-outputLayer[j]->o_value)*hiddenLayer[i]->weight[j];
        }
        //参照公式b1
        hiddenLayer[i]->bDeltaSum+=(sum*hiddenLayer[i]->o_value*(1-hiddenLayer[i]->o_value));
     }

     /*
      *求输入层对隐藏层的加权变化
      */
      for(int i=0;i<INNODE;i++){
         //从公式b1和w1可以看出 两个公式是有公因式 所以这部分代码相同
         double sum=0;
         for(int j=0;j<HIDENODE;j++){
            for(int k=0;k<OUTNODE;k++){
                sum+=(-1)*(outputLayer[k]->rightout-outputLayer[k]->o_value)*outputLayer[k]->o_value*(1-outputLayer[k]->o_value)*hiddenLayer[j]->weight[k];
            }
            //参照公式w1
            inputLayer[i]->wDeltaSum[j]+=(sum*hiddenLayer[j]->o_value*(1-hiddenLayer[j]->o_value)*inputLayer[i]->value);
         }
      }

}


/*
* 进行训练 （注意在修改加权和偏移时都是）
*/
void BpNet::doTraining(vector<Sample> sampleGroup, double threshold,int mostTimes){
    int sampleNum = sampleGroup.size();
    int trainTimes=0;
    bool isSuccess=true;

    while(error >= threshold){
        //判断是否超过最大训练次数
        if(trainTimes>mostTimes){
            isSuccess=false;
            break;
        }

        cout<<"训练次数:"<<trainTimes++<<"\t\t"<<"当前误差: " << error << endl;
        error = 0.f;

        //初始化输入层加权的delta和
        for (int i = 0; i < INNODE; i++){
            inputLayer[i]->wDeltaSum.assign(inputLayer[i]->wDeltaSum.size(), 0.f);
        }

        //初始化隐藏层加权和偏移的delta和
        for (int i = 0; i < HIDENODE; i++){
            hiddenLayer[i]->wDeltaSum.assign(hiddenLayer[i]->wDeltaSum.size(), 0.f);
            hiddenLayer[i]->bDeltaSum = 0.f;
        }

        //初始化输出层的偏移和
        for (int i = 0; i < OUTNODE; i++){
            outputLayer[i]->bDeltaSum = 0.f;
        }

        //完成所有样本的调用与反馈
        for (int iter = 0; iter < sampleNum; iter++){
            setInValue(sampleGroup[iter].in);
            setOutRightValue(sampleGroup[iter].out);

            fp();
            bp();
        }

        //修改输入层的加权
        for (int i = 0; i < INNODE; i++){
            for (int j = 0; j < HIDENODE; j++){
                //每一个加权的和都是所有样本累积的 所以要除以样本数
                inputLayer[i]->weight[j] -= LEARNINGRATE * inputLayer[i]->wDeltaSum[j] / sampleNum;
            }
        }

        //修改隐藏层的加权和偏移
        for (int i = 0; i < HIDENODE; i++){
            //修改每个节点的偏移 因为一个节点就一个偏移 所以不用在节点里再遍历
            hiddenLayer[i]->bias -= LEARNINGRATE * hiddenLayer[i]->bDeltaSum / sampleNum;

            //修改每个节点的各个加权的值
            for (int j = 0; j < OUTNODE; j++){
                hiddenLayer[i]->weight[j] -= LEARNINGRATE * hiddenLayer[i]->wDeltaSum[j] / sampleNum;
            }
        }

        //修改输出层的偏移
        for (int i = 0; i < OUTNODE; i++){
            outputLayer[i]->bias -= LEARNINGRATE * outputLayer[i]->bDeltaSum / sampleNum;
        }
    }

    if(isSuccess){
        cout<<endl<<"训练成功!!!"<<"\t\t"<<"最终误差: "<<error<<endl<<endl;
    }else{
        cout<<endl<<"训练失败! 超过最大次数!"<<"\t\t"<<"最终误差: "<<error<<endl<<endl;
    }

}


/*
* 训练后进行测试使用
*/
void BpNet::afterTrainTest(vector<Sample>& testGroup){
    int testNum = testGroup.size();

    for (int iter = 0; iter < testNum; iter++){
        //把样本输出清空
        testGroup[iter].out.clear();
        setInValue(testGroup[iter].in);

        //从隐藏层从输入层获取数据
        for (int i = 0; i < HIDENODE; i++){
            double sum = 0.f;
            for (int j = 0; j < INNODE; j++){
                sum += inputLayer[j]->value * inputLayer[j]->weight[i];
            }

            sum += hiddenLayer[i]->bias;
            hiddenLayer[i]->o_value = sigmoid(sum);
        }

        //输出层从隐藏层获取数据
        for (int i = 0; i < OUTNODE; i++){
            double sum = 0.f;
            for (int j = 0; j < HIDENODE; j++){
                sum += hiddenLayer[j]->o_value * hiddenLayer[j]->weight[i];
            }

            sum += outputLayer[i]->bias;
            outputLayer[i]->o_value = sigmoid(sum);

            //设置输出的值
            testGroup[iter].out.push_back(outputLayer[i]->o_value);
        }
    }
}


/*
* 给输入层每个节点设置输入值 每个样本进行训练时都要调用
*/
void BpNet::setInValue(vector<double> inValue){
    //对应一次样本 输入层每个节点的输入值
    for (int i = 0; i < INNODE; i++){
        inputLayer[i]->value = inValue[i];
    }
}

/*
* 给输出层每个节点设置正确值 每个样本进行训练时都要调用
*/
void BpNet::setOutRightValue(vector<double> outRightValue){
    //对应一次样本 输出层层每个节点的正确值
    for (int i = 0; i < OUTNODE; i++){
            outputLayer[i]->rightout = outRightValue[i];
    }
}

