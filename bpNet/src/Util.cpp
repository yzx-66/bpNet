#include "Util.h"
#include <string>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <vector>

using namespace std;

vector<double> Util::getFileData(char* fileName){
    vector<double> res;

    ifstream input(fileName);
    if(!input){
        return res;
    }

    string buff;
    while(getline(input,buff)){
        char* datas=(char*)buff.c_str();
        const char* spilt=" ";
        //strtok字符串拆分函数
        char* data=strtok(datas,spilt);

        while(data!=NULL){
            res.push_back(atof(data));
            //NULL代表从上次没拆分完地方继续拆
            data=strtok(NULL,spilt);
        }
    }

    input.close();
    return res;
}
