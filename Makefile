init:
	conda install -y pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch
	conda install -y -c conda-forge matplotlib
	python setup.py install

test:
	nosetests tests