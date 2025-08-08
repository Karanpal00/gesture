#include <bits/stdc++.h>
#include "inputParser.h"
#include "execute.h"
#include <filesystem>

using namespace std;

string getPathFromHome() {
    //find the path of direcory at start
    string str = filesystem::current_path();
    return str.substr(str.find_last_of('/')+1);
}

int main() {
    //shell loop
    while(true) {
        //prompt
        cout<<"~/"<<getPathFromHome()<<"/$ ";

        string input;
        getline(cin, input);
        
        //parse
        vector<string> parsedInput = parser(input);

        //execute
        cout<<executeCommand(parsedInput)<<endl;

    }

    return 0;
}