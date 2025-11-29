Following instruction in `PHC/docs/docker_instruction.MD`


-----------------------------

docker run -it --mount type=bind,source=$HOME/repos/ASE,target=/home/gymuser/ASE --network=host --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 hansen1416/phc /bin/bash