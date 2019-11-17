# Projeto MEC

## Introducao
Primeiramente, gostaria de agradecer ao EdjeElectronics por disponibilizar seu repositorio com a implementacao de deteccoes de objetos com TensorFlow a partir de uma Raspberry PI, esse projeto eh uma adaptacao desse projeto.
A finalidade desse sistema eh detectar quantas pessoas estao em algum ambiente a partir de uma Webcam (conectada ou via IP) e enviar essas informacoes para o Cloud Firestore do Firebase.

## Requisitos
- Projeto no Firebase criado
- Credenciais do Firebase que serao utilizadas para acessar o Cloud Firestore
- Raspberry PI 3 || Raspberry PI 4
- Webcam
- Raspbian SO

## Como usar
- Clone esse repositorio em qualquer pasta do sistema
- Copie seu arquivo de credenciais do Firebase para dentro da pasta database
- Acesse a pasta via terminal
- Crie e acesse um virtual environment com Python 3 (opcional)
- Execute o arquivo requirements.sh (bash requirements.sh)
- Execute: python3 TFLite_detection_webcam.py
- Algumas flags estao disponiveis para o comando acima, sendo elas:
  - --modeldir: Pasta em que o modelo .tflite esta localizado, OBRIGATORIO
  - --graph', Nome do arquivo .tflite, se diferente de detect.tflite
  - --labels', Nome do arquivo labelmap, se diferente de labelmap.txt
  - --threshold, Threshold de confianca minimo para deteccao dos objetos
