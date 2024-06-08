# FGVC Data Preparation
- We use the same splits as TOAST with the exception of Stanford Cars.
- Unfortunately, the data split (json files) are no longer available from the original source. We have included the splits we use in the data directories of each dataset.

### CUB200 2011
- Download the data from [http://www.vision.caltech.edu/visipedia/CUB-200-2011.html](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), and place in ```data/cub```.
- Preparation:
  ```sh
  cd data/cub
  tar -zxvf CUB_200_2011.tgz
  mv CUB_200_2011/images .
  ```
### NABirds
- Download the data from [http://info.allaboutbirds.org/nabirds/](http://info.allaboutbirds.org/nabirds/) and place in ```data/nabirds```.
- Preparation:
  ```sh
  cd data/nabirds
  unzip nabirds.zip
  ```
### Oxford Flowers
- Download the data from [https://www.robots.ox.ac.uk/~vgg/data/flowers/](https://www.robots.ox.ac.uk/~vgg/data/flowers/), and place in ```data/OxfordFlowers```.
- Preparation:
  ```sh
  cd data/oxfordflowers
  tar -zxvf oxfordflower102.tgz
  ```
### Stanford Dogs
- Download the data from [http://vision.stanford.edu/aditya86/ImageNetDogs/main.html](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html), and place in ```data/StanfordDogs```.
- Preparation:
  ```sh
  cd data/stanforddogs
  tar -xvf images.tar
  ```
### Stanford Cars
- For Stanford Cars, our splits is provided in the ```data/StanfordCars``` folder. The data can be downloaded from [here](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset). The data should be placed in the ```data/StanfordCars``` folder and unzipped.
