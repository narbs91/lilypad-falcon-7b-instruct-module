# Falcon-7b-instruct Lilypad module

Model: https://huggingface.co/tiiuae/falcon-7b-instruct

Docker image: https://hub.docker.com/r/narbs91/lilypad-falcon-7b-instruct-modulev8

# How to run

Run the Model

a) Run the model through lilypad
    ```
    lilypad --network demonet run github.com/narbs91/lilypad-falcon-7b-instruct-module:v1.8.0 --web3-private-key <your-private-key> -i MODEL_INPUT='Write me a haiku about Lilypads'
    ```

b) Run the model through bacalhau
    ```
    bacalhau docker run --api-host=bootstrap.demo.bacalhau.org --env MODEL_INPUT="Write me a haiku about Lilypads" --env HF_HUB_OFFLINE=1 narbs91/lilypad-falcon-7b-instruct-module:latest@sha256:ea0db2a579267f69e08f5fa9173a777f73580e8a2c18bcc4c803ebbfb233c304
    ```

c) Run the model through docker with HF_HUB_OFFLINE=1
    ```
    docker run -e HF_HUB_OFFLINE=1 -e MODEL_INPUT="Write a haiku about Lilypads" narbs91/lilypad-falcon-7b-instruct-module:latest
    ``` 

# Creating, tagging and pushing a new docker image
1. Build the docker image
    ```
    docker buildx build --platform linux/amd64 -t lilypad-falcon-7b-instruct-module:latest .
    ```

2. Tag the docker image
    ```
    docker tag lilypad-falcon-7b-instruct-module:latest <your-dockerhub-username>/lilypad-falcon-7b-instruct-module:latest
    ```

3. Push the docker image
    ```
    docker push <your-dockerhub-username>/lilypad-falcon-7b-instruct-module:latest
    ```

4. Create a new release on github