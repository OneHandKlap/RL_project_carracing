install_sys:
	sudo apt-get install -y libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb ffmpeg curl patchelf libglfw3 libglfw3-dev cmake zlib1g zlib1g-dev swig && conda env create -f environment.yml

install_dep:
	pip install -r requirements.txt

