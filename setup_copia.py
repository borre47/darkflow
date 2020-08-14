from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy
import os
import imp

path = __file__
path = path.split('/')[0]
print(path)
VERSION = imp.load_source('version', os.path.join(path, 'darkflow', 'version.py'))
VERSION = VERSION.__version__
print("{}.darkflow.cython_utils.nms".format(path))
if os.name =='nt' :
    ext_modules=[
        Extension("{}.darkflow.cython_utils.nms".format(path),
            sources=["{}/darkflow/cython_utils/nms.pyx".format(path)],
            #libraries=["m"] # Unix-like specific
            include_dirs=[numpy.get_include()]
        ),        
        Extension("{}.darkflow.cython_utils.cy_yolo2_findboxes".format(path),
            sources=["{}/darkflow/cython_utils/cy_yolo2_findboxes.pyx".format(path)],
            #libraries=["m"] # Unix-like specific
            include_dirs=[numpy.get_include()]
        ),
        Extension("{}.darkflow.cython_utils.cy_yolo_findboxes".format(path),
            sources=["{}/darkflow/cython_utils/cy_yolo_findboxes.pyx".format(path)],
            #libraries=["m"] # Unix-like specific
            include_dirs=[numpy.get_include()]
        )
    ]

elif os.name =='posix' :
    ext_modules=[
        Extension("{}.darkflow.cython_utils.nms".format(path),
            sources=["{}/darkflow/cython_utils/nms.pyx".format(path)],
            libraries=["m"], # Unix-like specific
            include_dirs=[numpy.get_include()]
        ),        
        Extension("{}.darkflow.cython_utils.cy_yolo2_findboxes".format(path),
            sources=["{}/darkflow/cython_utils/cy_yolo2_findboxes.pyx".format(path)],
            libraries=["m"], # Unix-like specific
            include_dirs=[numpy.get_include()]
        ),
        Extension("{}.darkflow.cython_utils.cy_yolo_findboxes".format(path),
            sources=["{}/darkflow/cython_utils/cy_yolo_findboxes.pyx".format(path)],
            libraries=["m"], # Unix-like specific
            include_dirs=[numpy.get_include()]
        )
    ]

else :
    ext_modules=[
        Extension("{}.darkflow.cython_utils.nms".format(path),
            sources=["{}/darkflow/cython_utils/nms.pyx".format(path)],
            libraries=["m"] # Unix-like specific
        ),        
        Extension("{}.darkflow.cython_utils.cy_yolo2_findboxes".format(path),
            sources=["{}/darkflow/cython_utils/cy_yolo2_findboxes.pyx".format(path)],
            libraries=["m"] # Unix-like specific
        ),
        Extension("{}.darkflow.cython_utils.cy_yolo_findboxes".format(path),
            sources=["{}/darkflow/cython_utils/cy_yolo_findboxes.pyx".format(path)],
            libraries=["m"] # Unix-like specific
        )
    ]

setup(
    version=VERSION,
	name='darkflow',
    description='Darkflow',
    license='GPLv3',
    url='https://github.com/thtrieu/darkflow',
    packages = find_packages(),
	scripts = ['flow'],
    ext_modules = cythonize(ext_modules)
)