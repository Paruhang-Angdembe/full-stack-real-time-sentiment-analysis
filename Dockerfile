FROM python:3.8-slim


# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt


ENTRYPOINT ["python3"]
CMD ["app.py"]