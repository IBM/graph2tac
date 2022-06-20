#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <vector>
#include <iostream>

template <typename T> struct select_npy_type {
};

template <> struct select_npy_type<uint64_t> {
  const static NPY_TYPES type = NPY_UINT64;
};

#ifdef __clang__
#pragma message  "NOT GNU"
template <> struct select_npy_type<unsigned long> {
  const static NPY_TYPES type = NPY_ULONG;
};
#endif


template <> struct select_npy_type<uint32_t> {
    const static NPY_TYPES type = NPY_UINT32;
};

template <> struct select_npy_type<bool> {
    const static NPY_TYPES type = NPY_BOOL;
};


template <typename T>
static PyObject * numpy_ndarray1d(const std::vector<T>&);

template <typename T>
static PyObject * __numpy_ndarray2d(const std::vector<std::vector<T>>&);

template <typename T, size_t N>
static PyObject * __numpy_ndarray2d(const std::vector<std::array<T, N>>&);



template <typename T>
static PyObject *
numpy_ndarray1d(const std::vector<T> &a) {
  npy_intp dims[1] = {(npy_intp)a.size()};
  PyObject *res = PyArray_SimpleNew(1, dims, select_npy_type<T>::type);
  T *res_data = (T *)PyArray_DATA((PyArrayObject *)res);
  std::copy(a.begin(), a.end(), res_data);
  return res;
}




template <typename T, size_t N>
static PyObject *
numpy_ndarray2d(const std::vector<std::array<T, N>> &a) {
    npy_intp dims[2] = {static_cast<npy_intp>(a.size()), N};
    PyObject *result = PyArray_SimpleNew(2, dims, select_npy_type<T>::type);
    T *result_data = (T *) PyArray_DATA((PyArrayObject *)result);
    for (size_t i = 0; i < a.size(); ++i) {
	std::copy(a[i].begin(), a[i].end(), result_data + i * N);
    }
    return result;
}


template <typename T>
static PyObject *
numpy_ndarray2d(const std::vector<std::pair<T, T>> &a) {
    npy_intp dims[2] = {static_cast<npy_intp>(a.size()), 2};
    PyObject *result = PyArray_SimpleNew(2, dims, select_npy_type<T>::type);
    T *result_data = (T *) PyArray_DATA((PyArrayObject *)result);
    for (size_t i = 0; i < a.size(); ++i) {
	*(result_data + 2 * i)  = a[i].first;
	*(result_data + 2 * i  + 1) = a[i].second;
    }
    return result;
}


template <typename T>
static PyObject *
numpy_ndarray2d(const std::map<T, T> &a) {
    npy_intp dims[2] = {static_cast<npy_intp>(a.size()), 2};
    PyObject *result = PyArray_SimpleNew(2, dims, select_npy_type<T>::type);
    T *result_data = (T *) PyArray_DATA((PyArrayObject *)result);
    size_t i = 0;
    for (const auto [k, v]: a) {
	*(result_data + 2 * i) = k;
	*(result_data + 2 * i + 1) = v;
	++i;
    }
    return result;
}




static PyObject * PyList_FromVector(const std::vector<PyObject*> vect) {
    PyObject * py_list = PyList_New(static_cast<Py_ssize_t>(vect.size()));
    for (size_t idx = 0; idx < vect.size(); ++idx) {
	PyList_SET_ITEM(py_list, idx, vect[idx]);
    }
    return py_list;
}

static PyObject * PyListBytes_FromVectorString(const std::vector<std::string> vect) {
    PyObject * py_list = PyList_New(static_cast<Py_ssize_t>(vect.size()));
    for (size_t idx = 0; idx < vect.size(); ++idx) {
	PyList_SET_ITEM(py_list, idx, Py_BuildValue("y", vect[idx].c_str()));
    }
    return py_list;
}



std::string PyBytes_AsSTDString(PyObject *bytesObj) {
  auto b_size = PyBytes_Size(bytesObj);
  char *buf = PyBytes_AsString(bytesObj);
  std::string s(buf, b_size);
  return s;
}


std::vector<std::string> PyListBytes_AsVectorString(PyObject *listObj) {
  if (!PyList_Check(listObj)) {
    throw std::invalid_argument("expected list input");
  }
  auto l_size = PyList_Size(listObj);
  std::vector<std::string> items;
  for (Py_ssize_t i = 0; i < l_size; ++i) {
    PyObject *Obj = PyList_GetItem(listObj, i);
    if (!PyBytes_Check(Obj)) {
      throw std::invalid_argument("expected bytes in arg " + std::to_string(i));
    }
    items.push_back(PyBytes_AsSTDString(Obj));
  }
  return items;
}

std::vector<uint64_t> PyArray_AsVector_uint64_t( PyArrayObject *  p_PyArrayObject) {
    uint64_t * p_array_data = (uint64_t *) PyArray_DATA(p_PyArrayObject);
    npy_intp array_size  = PyArray_SIZE(p_PyArrayObject);
    return std::vector<uint64_t> (p_array_data,
				  p_array_data + array_size);

}

std::vector<uint32_t> PyArray_AsVector_uint32_t( PyArrayObject *  p_PyArrayObject) {
    uint32_t * p_array_data = (uint32_t *) PyArray_DATA(p_PyArrayObject);
    npy_intp array_size  = PyArray_SIZE(p_PyArrayObject);
    return std::vector<uint32_t> (p_array_data,
				  p_array_data + array_size);

}


std::vector<uint8_t> PyArray_AsVector_uint8_t( PyArrayObject *  p_PyArrayObject) {
    uint8_t * p_array_data = (uint8_t *) PyArray_DATA(p_PyArrayObject);
    npy_intp array_size  = PyArray_SIZE(p_PyArrayObject);
    return std::vector<uint8_t> (p_array_data,
				  p_array_data + array_size);

}

std::vector<double> PyArray_AsVector_double( PyArrayObject *  p_PyArrayObject) {
    double * p_array_data = (double *) PyArray_DATA(p_PyArrayObject);
    npy_intp array_size  = PyArray_SIZE(p_PyArrayObject);
    return std::vector<double> (p_array_data,
				  p_array_data + array_size);
}




std::vector<std::pair<uint32_t, uint32_t>> Py2dArray_AsVector_uint32_pair( PyArrayObject *  p_PyArrayObject) {
    uint32_t * p_array_data = (uint32_t *) PyArray_DATA(p_PyArrayObject);
    npy_intp array_dim0  = PyArray_SIZE(p_PyArrayObject) / 2;
    std::vector<std::pair<uint32_t, uint32_t>> result;
    for (npy_intp idx = 0; idx < array_dim0; ++idx) {
	result.emplace_back(std::make_pair(*(p_array_data + 2*idx), *(p_array_data + 2*idx + 1)));
    }
    return result;
}




std::vector<std::vector<uint64_t>> PyListArray_AsVectorVector_uint64_t(PyObject *listObj) {
  if (!PyList_Check(listObj)) {
    throw std::invalid_argument("expected list input");
  }
  auto l_size = PyList_Size(listObj);

  std::vector<std::vector<uint64_t>> items;
  for (Py_ssize_t i = 0; i < l_size; ++i) {
      PyArrayObject* Obj = (PyArrayObject*) (PyList_GetItem(listObj, i));
    if (!PyArray_Check(Obj)) {
      throw std::invalid_argument("expected array in list element at position " + std::to_string(i));
    }

    items.push_back(PyArray_AsVector_uint64_t(Obj));
  }
  return items;

}

std::vector<std::vector<uint32_t>>  PyListArray_AsVectorVector_uint32_t(PyObject *listObj) {
  if (!PyList_Check(listObj)) {
    throw std::invalid_argument("expected list input");
  }
  auto l_size = PyList_Size(listObj);

  std::vector<std::vector<uint32_t>> items;
  for (Py_ssize_t i = 0; i < l_size; ++i) {
      PyArrayObject* Obj = (PyArrayObject*) (PyList_GetItem(listObj, i));
    if (!PyArray_Check(Obj)) {
      throw std::invalid_argument("expected array in list element at position " + std::to_string(i));
    }

    items.push_back(PyArray_AsVector_uint32_t(Obj));
  }
  return items;

}



std::vector<std::vector<std::pair<uint32_t, uint32_t>>> PyList2dArray_AsVectorVector_uint32_pair(PyObject *listObj) {
  if (!PyList_Check(listObj)) {
    throw std::invalid_argument("expected list input");
  }
  auto l_size = PyList_Size(listObj);

  std::vector<std::vector<std::pair<uint32_t, uint32_t>>> items;

  for (Py_ssize_t i = 0; i < l_size; ++i) {
      PyArrayObject* Obj = (PyArrayObject*) (PyList_GetItem(listObj, i));
    if (!PyArray_Check(Obj)) {
      throw std::invalid_argument("expected array in list element at position " + std::to_string(i));
    }

    if (PyArray_NDIM(Obj) != 2) {
	throw std::invalid_argument("expected 2d array in list element at position "
				    + std::to_string(i) + " but received " + std::to_string(PyArray_NDIM(Obj)) +
				    " dim array ");
    }
    if (*(PyArray_DIMS(Obj) + 1) !=2 ) {
	throw std::invalid_argument("expected 2d array in list element at position of shape (*, 2) "
				    + std::to_string(i) + " but received (*, " + std::to_string(*(PyArray_DIMS(Obj) + 1))
				    +  " )-shaped nparray ");
    }

    items.push_back(Py2dArray_AsVector_uint32_pair(Obj));
  }
  return items;


}
