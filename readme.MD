# Falcon-7b-instruct Lilypad module

Model:https://huggingface.co/tiiuae/falcon-7b-instruct

# How to run
1. Build the docker image
    ```
    docker build --platform linux/arm64 -t lilypad-falcon-7b-instruct-module:latest .
    ```

2. Run the model through docker
    ```
    docker run -e MODEL_INPUT="Write me a haiku about Lilypads" narbs91/lilypad-falcon-7b-instruct-module:latest
    ```