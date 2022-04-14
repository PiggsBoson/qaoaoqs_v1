from setuptools import setup, find_packages

VERSION = '1.0.0' 
DESCRIPTION = 'QAOA contorl on open quantum system '
LONG_DESCRIPTION = 'Completely resesigned with better OOP structure'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="qaoaoqs", 
        version=VERSION,
        author="Zhibo Yang",
        author_email="not_disclosed@mail.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['numpy', 'qutip'], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'first package'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)