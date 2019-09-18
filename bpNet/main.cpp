#include "BpNet.h"
#include "Util.h"

using namespace std;

void getInput(double& threshold,int& mostTimes);//获得输入的阀值和误差大小
vector<Sample> getTrianData();//从文件获取训练数据 没获取到直接退出
vector<Sample> getTestData();//从文件获取测试数据 没获取到直接退出
void showTest(vector<Sample>testGroup);//输出测试数据的结果

int main(){
    //准备所有数据
    BpNet bpNet;
    vector<Sample> sampleGroup=getTrianData();
    vector<Sample> testGroup=getTestData();
	double threshold;//设定的阀值
    int mostTimes;//最大训练次数

    //获取输入 并提示数据已经录入
    getInput(threshold,mostTimes);

    //进行训练
    bpNet.doTraining(sampleGroup,threshold,mostTimes);

    //训练后测试录入的数据 这里的参数是引用
    bpNet.afterTrainTest(testGroup);
    //打印提前录入数据的测试结果
    showTest(testGroup);

    return 0;
}

void getInput(double& threshold,int& mostTimes){
    cout<<"训练及测试数据已从文件读入"<<endl<<endl;
    cout<<"请输入XOR训练最大误差：";//0.0001最好
    cin>>threshold;
    cout<<"请输入XOR训练最大次数：";
    cin>>mostTimes;
}

void showTest(vector<Sample> testGroup){
    //输出测试结果
    cout<<"系统测试数据:"<<endl;
    for (int i=0;i<testGroup.size();i++){
        for (int j=0;j<testGroup[i].in.size();j++){
            cout<<testGroup[i].in[j]<<"\t";
        }

        cout<<"-- XOR训练结果 :";
        for (int j=0;j<testGroup[i].out.size();j++){
            cout << testGroup[i].out[j] << "\t";
        }
        cout<<endl;
    }

    cout<<endl<<endl;
    system("pause");
}


vector<Sample> getTestData(){
    Util util;
    vector<double> testData=util.getFileData("test.txt");
    if(testData.size()==0){
        cout<<"载入测试数据失败！"<<endl;
        exit(0);
    }

    int groups=testData.size()/2;
    //创建测试数据
    Sample testInOut[groups];

    for (int i = 0,index=0; i < groups; i++){
        for(int j=0;j<2;j++){
            testInOut[i].in.push_back(testData[index++]);
        }
    }

    //初始化数据
    return vector<Sample>(testInOut,testInOut+groups);
}

vector<Sample> getTrianData(){
    Util util;
    vector<double> trainData=util.getFileData("data.txt");
    if(trainData.size()==0){
        cout<<"载入训练数据失败！"<<endl;
        exit(0);
    }

    int groups=trainData.size()/3;
    //创建样本数据
    Sample trainInOut[groups];

    //把vector设置给样本Sample
    for (int i = 0,index=0; i < groups; i++){
        for(int j=0;j<3;j++){
            if(j%3!=2){
                trainInOut[i].in.push_back(trainData[index++]);
            }else{
                trainInOut[i].out.push_back(trainData[index++]);
            }
        }
    }

    //初始化录入的个数据
    return vector<Sample>(trainInOut,trainInOut+groups);
}
