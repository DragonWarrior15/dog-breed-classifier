# use tabs for indentation when writing recipes

.Phony: clean .FORCE get_data

.Force: # a dummy file to force build

get_data:
	mkdir data
	cd data
	wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
	unzip dogImages.zip
	cd ..

setup_env:
	conda create -n dog-breed-classifier-env
	conda activate dog-breed-classifier-env
	conda install pytorch torchvision torchaudio cpuonly -c pytorch
	conda install pandas
