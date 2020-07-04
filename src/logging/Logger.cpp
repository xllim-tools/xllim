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

void Logger::log(const std::string & msg) {
    if(pModule){
        CPyObject pFunc = PyUnicode_FromString("log");
        if(pFunc){
            CPyObject pArgs = Py_BuildValue(msg.c_str(), msg.length());
            PyObject_CallMethodObjArgs(py_obj, pFunc, pArgs.getObject());
        }
        else
        {
            printf("ERROR: function get_D_dimension() \n");
        }
    }else{
        printf("ERROR: Module not imported, Can not call function get_D_dimension()\n");
    };
}