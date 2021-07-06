#IMAGE=supervisely/base-py
#docker pull $IMAGE && \
#docker build --build-arg IMAGE=$IMAGE -t $IMAGE"-debug" . && \
#docker run --rm -it -p 7777:22 --shm-size='1G' -e PYTHONUNBUFFERED='1' $IMAGE"-debug"
# -v ~/max:/workdir

docker build -t supervisely/mmdetection3d:1.0.0 . && \
cp /root/.ssh/authorized_keys . && \
docker-compose build && \
docker-compose up -d && \
docker-compose ps
