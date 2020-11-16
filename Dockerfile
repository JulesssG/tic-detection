# For finding latest versions of the base image see
# https://github.com/SwissDataScienceCenter/renkulab-docker
ARG RENKU_BASE_IMAGE=renku/renkulab-cuda10.0-tf1.14:0.7.2
FROM ${RENKU_BASE_IMAGE}

# RENKU_VERSION determines the version of the renku CLI
# that will be used in this image. To find the latest version,
# visit https://pypi.org/project/renku/#history.

ARG RENKU_VERSION=0.11.6

# Uncomment and adapt if your R or python packages require extra linux (ubuntu) software
# e.g. the following installs apt-utils and vim; each pkg on its own line, all lines
# except for the last end with backslash '\' to continue the RUN line
#
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    apt-utils \
    vim \
    libgl1-mesa-glx \
    nvidia-cuda-toolkit \
    curl
USER ${NB_USER}

# install the python dependencies
COPY requirements.txt environment.yml /tmp/
RUN conda env update -q -f /tmp/environment.yml && \
    /opt/conda/bin/pip install -r /tmp/requirements.txt && \
    conda clean -y --all && \
    conda env export -n "root" && \
    jupyter labextension install @ryantam626/jupyterlab_vim && \
    jupyter labextension install @wallneradam/trailing_space_remover && \
    echo "alias g='git'\nalias gs='git status -s -b'\nalias gac='git add . && git commit'\nalias ga='git add'\nalias gc='git commit'\nalias gd='git diff'\nalias gb='git branch'\nalias gpl='git pull'\nalias gp='git push'\nalias gl='git --no-pager log -10 --oneline --format=format:'%C(bold blue)%h%C(reset) - %C(bold green)(%ar)%C(reset) %C(white)%s%C(reset) -%C(red) %an%C(reset)%C(bold yellow)%+d%C(reset)''\nalias gll='git log --oneline --graph --format=format:'%C(bold blue)%h%C(reset) - %C(bold green)(%ar)%C(reset) %C(white)%s%C(reset) -%C(red) %an%C(reset)%C(bold yellow)%+d%C(reset)''\nalias v='vim'\nalias ..='cd ..'\nalias -='cd -'\nalias gaddremote='git add remote gh https://github.com/JulesssG/tourette-tic-detector.git'" >> $HOME/.bashrc

########################################################
# Do not edit this section and do not add anything below
RUN pipx install --force renku==${RENKU_VERSION}
########################################################
