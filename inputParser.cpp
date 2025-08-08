#include "inputParser.h"
#include<sstream>
using namespace std;

vector<string> parser(string s) {
    stringstream ss(s);

    string word;
    vector<string> inputParsed;

    while(ss>>word){
        inputParsed.push_back(word);
    }

    return inputParsed;
}

