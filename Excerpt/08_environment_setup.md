# Environment Setup

The project is executed on [Ubuntu 23.04][ubuntu] in a python virtual environment, where the packages installed only affects the said environment. The virtual environment [Virtualenv][virtualenv] was used. This approach create a folder in root folder that will house the python binary and installed packages. Below will be hte steps to create a virtual environment and install the necessary packages.

- On [Ubuntu 23.04][ubuntu], [Virtualenv][virtualenv] can be installed using `sudo apt install virtualenv`. - Create the virtual environment in the root folder by running the command, `virtualenv venv`. where `venv` will be the name of the folder to hold the python binaries and later, packages that would be installed.
- Activate the virtual environment by running, `source venv/bin/activate`.
- In this project the packages used were written into a file, `requirement.txt`, using `pip freeze > requirements.txt`.
- Install the packages by running, `pip install -r requirements.txt`
- To deactivate, run, `deactivate` in the terminal.

Read more on how to use Virtual environments [here][virtualenv].

#

[ubuntu]: https://releases.ubuntu.com/lunar/
[virtualenv]: https://virtualenv.pypa.io/en/latest/
[python-virtual-env]: https://dev.to/otumianempire/python-virtual-environment-27ak
