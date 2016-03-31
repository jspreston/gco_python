gco_python: gco_src
	python setup.py build

gco-v3.0.zip:
	wget http://vision.csd.uwo.ca/code/gco-v3.0.zip

gco_src: gco-v3.0.zip
	mkdir gco_src
	cd gco_src && unzip ../gco-v3.0.zip

patch: gco_src
	patch gco_src/GCoptimization.h gco_exception.patch

setup: patch

install: gco_python
	python setup.py install

clean:
	rm -rf build gco_python.cpp

clobber: clean
	rm -rf gco_src gco-v3.0.zip
