# PocketDerma
![WhatsApp Image 2024-05-24 at 15 32 41](https://github.com/ChaseNaidoo/PocketDerma/assets/30260269/62064f8a-a4d8-484e-a7a3-ec9daa828e68)


PocketDerma aims to address the challenge of providing an easy and effective way for individuals to determine the health of their skin
through AI-driven disease detection. This web application allows users to take pictures of various skin diseases and match them with diseases in our
database using AI components, enabling users to get a diagnosis without much hassle.

## Project Description
For most people, getting access to good healthcare is not easy due to long distances and other factors. Our project aims to address this issue
by bringing healthcare to users through an application.This application allows users to take pictures of various skin diseases and match them 
with diseases in our database using AI components. This enables users to get a diagnosis without much hassle.

## Installation
To install and set up the project, follow these steps:
- Clone the repository: git clone https://github.com/ChaseNaidoo/PocketDerma.git
- Install dependencies: npm install

## Usage
    1. cd skin_disease_model/frontend
    2. Start the server: npm run start
    3. Open your browser and navigate to http://localhost:3000
    4. In another terminal, cd skin_disease_model/api
    5. Run ./main.py
    6. To train a custom model, modify training_model/datasets to include your datasets
    7. Modify main.py in skin_disease_model/api to include class names
    8. Modify training_model.py in training to include the correct number of classes
    9. cd skin_disease_model/training
    10. Run ./training_model.py
    11. MySQL should be running in another terminal to create an account and login

## Screenshots
### Login Page
![WhatsApp Image 2024-05-24 at 15 32 41](https://github.com/ChaseNaidoo/PocketDerma/assets/30260269/0d027c21-6148-4c41-b9bb-daadd88624ae)

### Signup Page
![WhatsApp Image 2024-05-24 at 17 31 54](https://github.com/ChaseNaidoo/PocketDerma/assets/30260269/84d78ea8-53de-4801-bf77-7c6550d24237)

### Image Upload
![Screenshot 2024-05-20 171751](https://github.com/ChaseNaidoo/PocketDerma/assets/125469506/d2e1763b-056f-463c-900a-227f75a3f1f0)

### Processed Image
![Screenshot 2024-05-20 at 17-21-30 PocketDerma](https://github.com/ChaseNaidoo/PocketDerma/assets/125469506/545acb3f-69b7-4631-b72e-4e7ea65befb7)

## Authors
- Cameron Chase Naidoo
- Alice Musukwa
- Gold Israel

## Licensing
This project is licensed under the MIT License.

