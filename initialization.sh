#! /bin/bash
echo -e "\033[32m acquiring privilege \033[0m"
sudo ls

echo -e "\033[32m inflating isaacgym \033[0m"
if md5sum Isaac*.tar.gz | grep 3677be2095b36a714fd2606f4438efe1
then
 echo IsaacGym package checked
else
 echo there\'s no proper IsaacGym package file
 exit
fi
rm -rf isaacgym
tar --totals -xf Isaac*.tar.gz
cd isaacgym
sed -i '620i\    return sample' python/isaacgym/gymutil.py

echo -e "\033[32m creating rlgpu python env \033[0m"
cat ~/.condarc
./create_conda_env_rlgpu.sh
CONDA_DIR="$(conda info --base)"
source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate rlgpu
cd ..

echo -e "\033[32m installing mujoco dependencies \033[0m"
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
sudo rm /usr/lib/x86_64-linux-gnu/libGL.so
sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so

echo -e "\033[32m downloading mujoco200 and key \033[0m"
wget https://www.roboti.us/download/mujoco200_linux.zip
wget https://www.roboti.us/file/mjkey.txt

echo -e "\033[32m installing mujoco200 \033[0m"
rm -rf ~/.mujoco
sudo apt install -y unzip
unzip mujoco*.zip
rm mujoco*.zip
mkdir ~/.mujoco
mv mjkey.txt ~/.mujoco
mv mujoco200_linux ~/.mujoco/mujoco200
if test $(echo $LD_LIBRARY_PATH | grep mujoco200)
then
 echo \$LD_LIBRARY_PATH is set
else
 echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin' >> ~/.bashrc
 . ~/.bashrc
fi

echo -e "\033[32m installing isaacgymenvs \033[0m"
cd ..
rm -rf isaacgymenvs
git clone git@github.com:Robot-Learning-Library/isaacgymenvs.git
cd isaacgymenvs
sudo apt install -y libpython3.7
pip install -e .
cd ../upesi

echo -e "\033[32m installing robolite \033[0m"
cd ..
rm -rf robolite
git clone https://github.com/quantumiracle/robolite.git
cd robolite
pip install -r requirements.txt
pip install -e .
cd ../upesi

echo -e "\033[32m installing requirements \033[0m"
pip install -r requirements.txt

rm -rf log
mkdir log