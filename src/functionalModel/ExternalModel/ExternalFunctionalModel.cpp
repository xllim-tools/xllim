/**
 * @file ExternalFunctionalModel.cpp
 * @author Sami DJOUADI
 * @version 1.2
 * @date 22/04/2020
 */

#include "ExternalFunctionalModel.h"
#include <cstdio>
#include <Python.h>
#include "numpy/arrayobject.h"
#include "pyhelper.hpp"

using namespace Functional;

// wrapper around macro that returns NULL
int call_numpy_import_array_macro(){
    import_array();
    return 0;
}

ExternalFunctionalModel::ExternalFunctionalModel(const std::string &className, const std::string &fileName, const std::string &filePath) {
    CPyObject sys_path = PySys_GetObject("path");

    PyList_Append(sys_path, PyUnicode_FromString(filePath.c_str()));
    CPyObject pName = PyUnicode_FromString(fileName.c_str());
    pModule = PyImport_Import(pName);
    if(!pModule){
        printf("ERROR: Module not imported\n");
    }else{
        CPyObject dict = PyModule_GetDict(pModule);
        CPyObject py_class = PyDict_GetItemString(dict, className.c_str());
        if(PyCallable_Check(py_class)){
            py_obj = PyObject_CallObject(py_class, NULL);
        }else{
            printf("ERROR: Class not found \n");
        }
    }
    call_numpy_import_array_macro();
}

void ExternalFunctionalModel::F(rowvec x, rowvec &y) {
    if(pModule){

        CPyObject pFunc = PyUnicode_FromString("F");

        if(pFunc){
            double *x_ptr = x.memptr();
            npy_intp x_dims[1]{get_L_dimension()};
            CPyObject x_pArray = PyArray_SimpleNewFromData(1, x_dims, NPY_DOUBLE, x_ptr);

            CPyObject y_pModified = PyObject_CallMethodObjArgs(py_obj, pFunc , x_pArray.getObject(), NULL);
            auto y_arrayModified = reinterpret_cast<PyArrayObject*>(y_pModified.getObject());

            y = rowvec(&reinterpret_cast<double*>(PyArray_DATA(y_arrayModified))[0], get_D_dimension());
        }
        else
        {
            printf("ERROR: function F(x,y) \n");
        }
    }else{
        printf("ERROR: Module not imported, Can not call function F(x,y)\n");
    }
}

int ExternalFunctionalModel::get_D_dimension() {
    if(pModule){
        CPyObject pFunc = PyUnicode_FromString("get_D_dimension");
        if(pFunc){
            CPyObject pValue = PyObject_CallMethodObjArgs(py_obj, pFunc, NULL);
            return (int)PyLong_AsLong(pValue);
        }
        else
        {
            printf("ERROR: function get_D_dimension() \n");
        }
    }else{
        printf("ERROR: Module not imported, Can not call function get_D_dimension()\n");
    };
    return 0;
}

int ExternalFunctionalModel::get_L_dimension() {
    if(pModule){
        CPyObject pFunc = PyUnicode_FromString("get_L_dimension");
        if(pFunc){
            CPyObject pValue = PyObject_CallMethodObjArgs(py_obj, pFunc, NULL);
            return (int)PyLong_AsLong(pValue);
        }
        else
        {
            printf("ERROR: function get_D_dimension() \n");
        }
    }else{
        printf("ERROR: Module not imported, Can not call function get_D_dimension()\n");
    };
    return 0;
}

void ExternalFunctionalModel::to_physic(rowvec &x) {
    if(pModule) {
        CPyObject pFunc = PyUnicode_FromString("to_physic");
        if(pFunc){

            double *x_ptr = x.memptr();
            npy_intp x_dims[1]{get_L_dimension()};
            CPyObject x_pArray = PyArray_SimpleNewFromData(1, x_dims, NPY_DOUBLE, x_ptr);


            CPyObject x_pModified = PyObject_CallMethodObjArgs(py_obj, pFunc, x_pArray.getObject(), NULL);
            auto x_arrayModified = reinterpret_cast<PyArrayObject*>(x_pModified.getObject());

            x = rowvec(&reinterpret_cast<double*>(PyArray_DATA(x_arrayModified))[0], get_L_dimension());
        }
        else
        {
            printf("ERROR: function to_physic(x) \n");
        }
    }else{
        printf("ERROR: Module not imported, Can not call function to_physic(x)\n");
    }
}

void ExternalFunctionalModel::to_physic(double *x, unsigned int size) {
    //TODO : implement
}

void ExternalFunctionalModel::from_physic(double *x, unsigned int size) {
    if(pModule) {
        CPyObject pFunc = PyUnicode_FromString("from_physic");
        if (pFunc) {

            npy_intp x_dims[1]{size};
            CPyObject x_pArray = PyArray_SimpleNewFromData(1, x_dims, NPY_DOUBLE, x);

            CPyObject x_pModified = PyObject_CallMethodObjArgs(py_obj, pFunc, x_pArray.getObject(), NULL);
            auto x_arrayModified = reinterpret_cast<PyArrayObject *>(x_pModified.getObject());

            auto *x_modified = reinterpret_cast<double *>(PyArray_DATA(x_arrayModified));

            for(unsigned int i=0; i<size ; i++){
                x[i] = x_modified[i];
            }

        } else {
            printf("ERROR: function from_physic(x) \n");
        }
    }else{
        printf("ERROR: Module not imported, Can not call function from_physic(x)\n");
    }
}
