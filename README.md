Chatbot System with Reformer Model

Welcome to the Chatbot System project! This project is designed to create an intelligent chatbot capable of understanding user queries, extracting relevant information from a structured database, and generating accurate responses. The chatbot is powered by a Reformer-based neural network, which is a state-of-the-art model for natural language processing tasks. Below, you'll find everything you need to set up, train, and interact with the chatbot.
Table of Contents

    Project Overview

    Features

    Requirements

    Installation

    Usage

    File Descriptions

    Training the Model

    Running the Chatbot

    Contributing

    License

Project Overview

This project is a Chatbot System that uses a Reformer model to understand and respond to user queries. The chatbot is trained on synthetic dialogue data generated from a structured database, slots, and ontology files. It can handle queries related to various domains such as hotels, restaurants, attractions, and more. The system is designed to be modular, with separate scripts for data preprocessing, model training, and chatbot interaction.
Features

    Natural Language Understanding: The chatbot uses a Reformer model to understand user queries and generate responses.

    Database Querying: It can extract relevant information from a structured database based on user input.

    Interactive Chat Interface: Users can interact with the chatbot in real-time.

    Customizable Training Data: The chatbot can be trained on custom datasets for specific domains.

    Modular Design: The project is divided into separate scripts for data preprocessing, model training, and chatbot interaction.

Requirements

To run this project, you'll need the following:

    Python 3.x

    TensorFlow (for model training and inference)

    SentencePiece (for tokenization)

    NumPy (for numerical operations)

    JSON (for data handling)

You can install the required libraries using the following command:
bash
Copy

pip install tensorflow sentencepiece numpy

Installation

    Clone the Repository:
    bash
    Copy

    git clone https://github.com/fatemefaraji/chatbot.git
    cd chatbot-reformer

    Download the Data:

        Place your database files (police_db.json, hotel_db.json, etc.) in the data folder.

        Ensure the following files are present:

            processed_database.json

            processed_slots.json

            processed_ontology.json

            tokenized_training_data.json

    Install Dependencies:
    bash
    Copy

    pip install -r requirements.txt

Usage
File Descriptions

    Preprocess.py: Preprocesses raw database files and extracts unique keys. Generates processed_database.json.

    processed_ontology.py: Processes the ontology file and organizes slot values by domain. Generates processed_ontology.json.

    processed_slots.py: Processes slot descriptions and organizes them by domain. Generates processed_slots.json.

    process1.py: Generates synthetic training data (question-answer pairs) for the chatbot. Generates chatbot_training_data.json.

    tokenize_script.py: Tokenizes the training data using SentencePiece and generates tokenized_training_data.json.

    reformer.py: Defines the Reformer model architecture.

    train.py: Trains the Reformer model on the tokenized training data.

    chatbot.py: Implements the chatbot interface for interacting with users.

Training the Model

    Preprocess the Data:

        Run Preprocess.py, processed_ontology.py, and processed_slots.py to generate the necessary JSON files.

        Run process1.py to generate synthetic training data.

        Run tokenize_script.py to tokenize the training data.

    Train the Model:

        Run train.py to train the Reformer model. The trained model will be saved as reformer_final_model.keras.

Running the Chatbot

    Start the Chatbot:

        Run chatbot.py to start the chatbot.

        The chatbot will prompt you to type your queries.

    Interact with the Chatbot:

        Type your queries, and the chatbot will respond based on the trained model and database.

        Example:
        Copy

        You: What is the name of the hotel?
        Assistant: I found 1 matching hotel(s):
        1. Grand Hotel (luxury) in downtown - expensive price range
           phone: 123-456-7890
           address: 123 Main St

    Exit the Chat:

        Type exit to end the chat session.

Contributing

We welcome contributions to this project! If you'd like to contribute, please follow these steps:

    Fork the repository.

    Create a new branch for your feature or bugfix.

    Commit your changes and push them to your fork.

    Submit a pull request with a detailed description of your changes.

License

This project is licensed under the MIT License. Feel free to use, modify, and distribute the code as per the license terms.
Acknowledgments

    Reformer Model: This project uses the Reformer architecture, which is based on the paper "Reformer: The Efficient Transformer".

    TensorFlow: The project leverages TensorFlow for model training and inference.

    SentencePiece: Used for tokenization of the training data.

Contact

If you have any questions or feedback, feel free to reach out:

    Email: lenafaraji.dev@gmail.com

    GitHub: fatemefaraji

Thank you for using the Chatbot System! We hope you find it useful and enjoy building intelligent chatbots with it.
