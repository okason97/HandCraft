FROM nvcr.io/nvidia/pytorch:24.05-py3

RUN apt-get update

# Create a working directory
WORKDIR /HandCraft

RUN mkdir -p /datasets/LSFB/

# Copy the current directory contents into the container at /workspace
COPY . /HandCraft


# Install Python dependencies if you have a requirements.txt file
RUN pip install --no-cache-dir -r requirements.txt

# Copy the dataset directory to the datasets folder
COPY /disco2/datasets/LSFB /datasets/LSFB/
# alternatively download and format the dataset using the following script
# LSFB setup script
# RUN python ./src/data/setup_lsfb.py -data_dir /datasets/LSFB/

# Set the entry point for the container
#ENTRYPOINT ["python"]
#CMD ["main.py"]

# wandb login