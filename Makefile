init:
	conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
	conda install -y -c conda-forge matplotlib
	python setup.py install

test:
	nosetests tests