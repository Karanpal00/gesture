#include "execute.h"
#include <sys/wait.h>
#include <unistd.h>
#include <iostream>
#include "cd.h"

string executeCommand(vector<string> input) {
    if (input.empty()) return "";

    if (input[0] == "cd") {
        if (input.size() == 1) return "Path NOt Provided..........";
        changeDir(input[1]);

        return"";

    }

    pid_t pid = fork();

    //child process
    if (pid == 0) {
        char* args [input.size()+1];

        for (size_t i = 0; i < input.size(); ++i) {
            args[i] = const_cast<char*>(input[i].c_str());
        }

        args[input.size()] = NULL;
        //execute in child process
        execvp(args[0], args);
        //if failed
        perror("execvp failed");
        exit(1);
    } else if (pid > 0) {
        //wait for child process
        int status;
        waitpid(pid, &status, 0);
    } else{
        perror("Fork failed");
    }

    return "";
}
