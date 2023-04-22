all: pre_clean build post_clean

post_clean:
	rm -fr _build

build: pre_clean
	tactician exec dune build
	sh ../package.sh

pre_clean:
	rm -fr _build
	rm -fr dataset
	tactician exec dune clean
