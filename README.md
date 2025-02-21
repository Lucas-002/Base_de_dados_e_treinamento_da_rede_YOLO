# Projeto de Treinamento da Rede YOLO

## Descrição
Este projeto tem como objetivo a criação de uma base de dados rotulada e o treinamento de um modelo YOLO para detecção de objetos. O treinamento pode ser realizado com imagens rotuladas manualmente ou utilizando o conjunto de dados COCO.

## Requisitos
Antes de iniciar, certifique-se de que possui os seguintes requisitos instalados:

- Python 3.8+
- OpenCV
- NumPy
- LabelMe (para rotulagem manual)
- Darknet (YOLO)
- Google Colab (opcional, para Transfer Learning)

### Instalação de Dependências
Execute o seguinte comando para instalar as bibliotecas necessárias:
```bash
pip install opencv-python numpy labelme
```

## Etapas do Projeto
### 1. Rotulagem de Imagens
Caso esteja criando sua própria base de dados, utilize o [LabelMe](http://labelme.csail.mit.edu/Release3.0/) para rotular as imagens.

1. Abra o LabelMe e carregue as imagens.
2. Desenhe as caixas delimitadoras para cada objeto.
3. Salve os arquivos no formato JSON ou TXT para uso no YOLO.

### 2. Configuração do YOLO
Baixe o YOLO em [Darknet](https://pjreddie.com/darknet/yolo/) e configure o ambiente:
```bash
git clone https://github.com/pjreddie/darknet.git
cd darknet
make
```
Edite o arquivo `cfg/yolov4.cfg` para ajustar os hiperparâmetros e definir as classes personalizadas.

### 3. Transfer Learning (Opcional)
Caso seu computador não suporte o treinamento completo, utilize o Google Colab:
1. Abra o [notebook Colab](https://colab.research.google.com/drive/1lTGZsfMaGUpBG4inDIQwIJVW476ibXk_#scrollTo=j0t221djS1Gk).
2. Faça upload dos seus dados rotulados.
3. Execute o código para realizar o treinamento.

### 4. Treinamento do Modelo
Execute o seguinte comando para iniciar o treinamento:
```bash
./darknet detector train data/obj.data cfg/yolov4.cfg yolov4.conv.137
```
Isso iniciará o processo de treinamento utilizando as imagens rotuladas.

### 5. Testando o Modelo
Para testar o modelo treinado, utilize o seguinte comando:
```bash
./darknet detector test data/obj.data cfg/yolov4.cfg backup/yolov4_final.weights image.jpg
```

## Resultado Esperado
Após o treinamento, o modelo será capaz de detectar pelo menos duas classes personalizadas além das classes pré-existentes no YOLO.



## Referências
- [LabelMe](http://labelme.csail.mit.edu/Release3.0/)
- [YOLO - Darknet](https://pjreddie.com/darknet/yolo/)
- [COCO Dataset](https://cocodataset.org/#home)
- [Google Colab Transfer Learning](https://colab.research.google.com/drive/1lTGZsfMaGUpBG4inDIQwIJVW476ibXk_#scrollTo=j0t221djS1Gk)

