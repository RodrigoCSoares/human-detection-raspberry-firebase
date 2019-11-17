# Projeto MEC
 
## Introdução
Primeiramente, gostaria de agradecer ao EdjeElectronics por disponibilizar seu repositório com a implementação de detecção de objetos com TensorFlow a partir de uma Raspberry PI, esse projeto é uma adaptação desse projeto.
A finalidade desse sistema é detectar quantas pessoas estão em algum ambiente a partir de uma Webcam (conectada ou via IP) e enviar essas informações para o Cloud Firestore do Firebase.
 
## Requisitos
- Projeto no Firebase criado
- Credenciais do Firebase que serão utilizadas para acessar o Cloud Firestore
- Raspberry PI 3 || Raspberry PI 4
- Webcam
- Raspbian SO
 
## Como usar
- Clone esse repositório em qualquer pasta do sistema
- Copie seu arquivo de credenciais do Firebase para dentro da pasta database
- Acesse a pasta via terminal
- Crie e acesse um virtual environment com Python 3 (opcional)
- Execute o arquivo requirements.sh (bash requirements.sh)
- Execute: python3 TFLite_detection_webcam.py
- Algumas flags estão disponíveis para o comando acima, sendo elas:
  - --modeldir: Pasta em que o modelo .tflite está localizado, OBRIGATÓRIO
  - --graph', Nome do arquivo .tflite, se diferente de detect.tflite
  - --labels', Nome do arquivo labelmap, se diferente de labelmap.txt
  - --threshold, Threshold de confiança mínimo para detecção dos objetos
  - --sleep, Configura o número de segundos entre as detecções
  - --cameraip, IP da câmera
  - --showlog, True para mostrar o log ou False para não mostrar
 

