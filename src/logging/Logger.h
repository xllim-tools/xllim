//
// Created by reverse-proxy on 4‏/7‏/2020.
//

#ifndef KERNELO_LOGGER_H
#define KERNELO_LOGGER_H

#include "../functionalModel/ExternalModel/pyhelper.hpp"
#include <Python.h>
#include <string>



namespace Logging{

    class Logger{
    private:
        CPyObject pModule;
        CPyObject py_obj;
    protected:
        Logger(){
            CPyObject pName = PyUnicode_FromString(std::string("kernelo").c_str());
            pModule = PyImport_Import(pName);
            if(!pModule){
                printf("ERROR: Module not imported\n");
            }else{
                CPyObject dict = PyModule_GetDict(pModule);
                CPyObject py_class = PyDict_GetItemString(dict, std::string("Logger").c_str());
                if(PyCallable_Check(py_class)){
                    py_obj = PyObject_CallObject(py_class, NULL);
                }else{
                    printf("ERROR: Class not found \n");
                }
            }
        }

        static Logger* logger_;

    public:

        Logger(Logger &other) = delete;
        void operator=(const Logger &) = delete;

        static Logger *GetInstance();

        void log(const std::string &msg);
    };

}

#endif //KERNELO_LOGGER_H
