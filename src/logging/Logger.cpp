//
// Created by reverse-proxy on 4‏/7‏/2020.
//


#include "Logger.h"
#include "Python.h"

using namespace Logging;

Logger* Logger::logger_ = nullptr;

std::string Logger::level(Level level){
    switch (level){
        case INFO:
            return "INFO";
        case WARNING:
            return "WARNING";
        case ERROR:
            return "ERROR";
        case CRITICAL:
            return "CRITICAL";
    }
}

Logger *Logger::GetInstance() {
    if(logger_ == nullptr){
        logger_ = new Logger();
    }
    return logger_;
}

void Logger::log(const std::string & msg, const std::string &level) {
    if(pModule){
        CPyObject pFunc = PyUnicode_FromString(std::string("log").c_str());
        if(pFunc){
            CPyObject pArgs_msg = PyUnicode_FromString(msg.c_str());
            CPyObject pArgs_level = PyUnicode_FromString(level.c_str());

            PyObject_CallMethodObjArgs(py_obj, pFunc, pArgs_msg.getObject(), pArgs_level.getObject(), NULL);
        }
        else
        {
            printf("ERROR: function log() \n");
        }
    }else{
        printf("ERROR: Module not imported, Can not call log()\n");
    };
}