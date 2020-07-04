//
// Created by reverse-proxy on 4‏/7‏/2020.
//


#include "Logger.h"
#include "Python.h"

using namespace Logging;

Logger *Logger::GetInstance() {
    if(logger_ == nullptr){
        logger_ = new Logger();
    }
    return logger_;
}

//void Logger::log(const std::string & msg) {
//    if(pModule){
//        CPyObject pFunc = PyUnicode_FromString(std::string("log").c_str());
//        if(pFunc){
//            CPyObject pArgs = Py_BuildValue(msg.c_str(), msg.length());
//            PyObject_CallMethodObjArgs(py_obj, pFunc, pArgs.getObject());
//        }
//        else
//        {
//            printf("ERROR: function log() \n");
//        }
//    }else{
//        printf("ERROR: Module not imported, Can not call log()\n");
//    };
//}