FROM nvcr.io/nvidia/pytorch:21.02-py3

RUN apt-get update

# Create a working directory
WORKDIR /HandCraft

# Copy the current directory contents into the container at /workspace
COPY . /HandCraft

# Install Python dependencies if you have a requirements.txt file
#COPY requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt

# LSFB setup script
RUN python ./src/data/setup_lsfb.py

# Set the entry point for the container
#ENTRYPOINT ["python"]
#CMD ["main.py"]
