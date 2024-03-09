FROM ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get dist-upgrade -y && \
    apt-get install libcurl4-openssl-dev \
    libssl-dev \
    libjansson-dev \
    automake \
    autotools-dev \
    build-essential \
    git \
    freeglut3 \
    freeglut3-dev \
    libxi-dev \
    libxmu-dev \
    wget \
    -y
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
RUN mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget -q https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
RUN apt-key add /var/cuda-repo-ubuntu2204-11-8-local/*.pub
RUN cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
RUN apt-get update
RUN apt-get -y install cuda
RUN git clone https://github.com/tpruvot/ccminer
WORKDIR ccminer
RUN git checkout linux
RUN ./build.sh
RUN ldconfig /usr/local/cuda/lib64
#RUN ./ccminer --version
RUN apt-get remove libcurl4-openssl-dev \
    libssl-dev \
    libjansson-dev \
    freeglut3 \
    freeglut3-dev \
    libxi-dev \
    libxmu-dev \
    -y
#ENTRYPOINT [ "./ccminer" ]
