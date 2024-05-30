FROM python:3.9-slim-buster as builder

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Update and install dependencies
RUN apt update -y && apt install awscli -y

# Install the Python dependencies
RUN pip install -r requirements.txt

# Download the spaCy language model(s)
RUN xargs -a spacy_requirements.txt -I {} python -m spacy download {}

# Final stage
FROM python:3.9-slim-buster

# Set the working directory
WORKDIR /app

# Copy only necessary files from the builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . /app

# Run the application
CMD ["python3", "app.py"]
