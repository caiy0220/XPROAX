wget --output-document resources.zip https://box.fu-berlin.de/s/Skm7oNiZnE6ZHFd/download
unzip resources.zip
mv resources/data .
mv resources/models .
mv resources/checkpoints generator
rm resources.zip
rm -rf resources
