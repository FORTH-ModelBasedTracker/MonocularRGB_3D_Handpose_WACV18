# Libraries for MonocularRGB 3D Handpose estimation (WACV18)

## Binaries for **Ubuntu 16.04**

Please download the libraries package from [here](http://cvrlcode.ics.forth.gr/files/wacv18/wacv18_libs_v1.0.tgz)
the files and place them in this folder.

There is a number of dependencies (ubuntu packages) you will need to install with apt:

```bash
sudo apt install libgoogle-glog-dev libtbb-dev libcholmod3.0.6 libatlas-base-dev libopenni0 libbulletdynamics2.83.6
```

## Environment

You must set your **LD_LIBRARY_PATH** to point also to this folder ie:

```bash
export LD_LIBRARY_PATH=$REPO_FOLDER/lib:$LD_LIBRARY_PATH
```

