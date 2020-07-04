//
// Created by reverse-proxy on 4‏/7‏/2020.
//


#include "Logger.h"
#include "Python.h"

using namespace Logging;

Logger* Logger::logger_ = nullptr;

Logger *Logger::GetInstance() {
    if(logger_ == nullptr){
        logger_ = new Logger();
    }
    return logger_;
}

void Logger::log(const std::string & msg) {
    if(pModule){
        CPyObject pFunc = PyUnicode_FromString(std::string("log").c_str());
        if(pFunc){
            CPyObject pArgs = PyUnicode_FromString(msg.c_str());
            PyObject_CallMethodObjArgs(py_obj, pFunc, pArgs.getObject(), NULL);
        }
        else
        {
            printf("ERROR: function log() \n");
        }
    }else{
        printf("ERROR: Module not imported, Can not call log()\n");
    };
}