# YOLO Contador de Pessoas

Uma aplicação Python que usa o modelo de deep learning YOLO (You Only Look Once) para detectar e contar pessoas em imagens. Este projeto fornece uma estrutura limpa e organizada para processamento em lote de imagens e geração de resultados anotados.

## Funcionalidades

- Detecção de Pessoas: Usa modelos YOLO11 para detectar pessoas em imagens
- Processamento em Lote: Processa múltiplas imagens de uma vez
- Alta Precisão: Thresholds de confiança e IoU configuráveis
- Estrutura Organizada: Separação limpa de amostras, resultados e modelos
- Suporte GPU: Aceleração CUDA para processamento mais rápido
- Interface de Linha de Comando: CLI fácil de usar com múltiplas opções
- Tratamento de Erros: Tratamento robusto de erros e saída informativa

## Estrutura do Projeto

```
yolo-count/
├── main.py              # Script principal da aplicação
├── config.py            # Configurações
├── samples/             # Diretório de imagens de entrada
│   └── .gitkeep        # Mantém diretório no git
├── results/             # Imagens anotadas de saída
│   └── .gitkeep        # Mantém diretório no git
├── models/              # Arquivos de modelo YOLO
│   └── .gitkeep        # Mantém diretório no git
├── pyproject.toml       # Dependências do projeto
├── uv.lock             # Arquivo de lock das dependências
├── requirements.txt     # Dependências para pip
├── .gitignore          # Regras do git ignore
└── README.md           # Este arquivo
```

## Instalação

### Pré-requisitos

- Python 3.8 ou superior
- GPU compatível com CUDA (opcional, para aceleração)

### Configuração

1. **Clone o repositório**:
   ```bash
   git clone https://github.com/gabrielcesar-dev/ai-projects.git
   cd ai-projects/yolo-count
   ```

2. **Instale as dependências usando uv**:
   ```bash
   uv sync
   ```

   Ou usando pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Os modelos YOLO serão baixados automaticamente no primeiro uso**

## Uso

### Uso Básico

1. **Coloque suas imagens** no diretório `samples/`
2. **Execute o script**:
   ```bash
   python main.py
   ```

### Opções de Linha de Comando

```bash
python main.py [OPÇÕES]

Opções:
  --model PATH        Caminho para o modelo YOLO
  --samples DIR       Diretório contendo imagens de amostra
  --results DIR       Diretório para salvar resultados
  --device DEVICE     Dispositivo para inferência
  --conf FLOAT        Threshold de confiança
  --iou FLOAT         Threshold IoU
  --show             Exibir imagens processadas
  --single FILE      Processar arquivo de imagem única
  --help             Mostrar mensagem de ajuda
```

## Configuração

As configurações padrão podem ser modificadas no arquivo `config.py`:

- `MODEL_PATH`: Caminho para o modelo YOLO
- `DEVICE`: Dispositivo para inferência ('cuda:0' ou 'cpu')
- `CONFIDENCE_THRESHOLD`: Threshold de confiança para detecção
- `IOU_THRESHOLD`: Threshold IoU para NMS
- `TARGET_SIZE`: Tamanho alvo para redimensionamento
- `SAMPLES_DIR`: Diretório de amostras
- `RESULTS_DIR`: Diretório de resultados

### Formatos de Imagem Suportados

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

## Informações do Modelo

Este projeto suporta modelos YOLO11:

- **yolo11n.pt**: Modelo Nano (mais rápido, menos preciso)
- **yolo11s.pt**: Modelo Small (balanceado velocidade/precisão) - Padrão
- **yolo11m.pt**: Modelo Medium (mais preciso, mais lento)
- **yolo11l.pt**: Modelo Large (alta precisão, lento)
- **yolo11x.pt**: Modelo Extra Large (maior precisão, mais lento)

## Saída

A aplicação gera:

1. **Imagens Anotadas**: Salvas no diretório `results/` com caixas delimitadoras e pontuações de confiança
2. **Saída do Console**: Contagens de detecção e resumo de processamento
3. **Resumo em Lote**: Contagem total de pessoas em todas as imagens processadas

## Licença YOLO

Este projeto utiliza os modelos YOLO da Ultralytics, que estão disponíveis sob diferentes opções de licenciamento:

### AGPL-3.0 (Licença Usada Neste Projeto)

- **Para quem**: Estudantes, pesquisadores e entusiastas
- **Tipo**: Licença open-source aprovada pela OSI
- **Requisitos**: Projetos que usam componentes AGPL-3.0 devem ser open-source
- **Uso comercial**: Permitido, mas deve manter o código aberto
- **Distribuição**: Código fonte deve estar disponível

### Outras Opções de Licenciamento

- **Ultralytics Enterprise**: Para uso comercial sem requisitos open-source
- **Ultralytics Academic**: Para universidades e instituições de pesquisa
- **Características**: Controle total, propriedade privada, suporte personalizado

### Importante

Se você pretende usar este projeto para fins comerciais e não deseja abrir o código fonte, considere adquirir uma licença Enterprise da Ultralytics.

Para mais informações sobre licenciamento, visite: [ultralytics.com/license](https://www.ultralytics.com/license)

## Agradecimentos

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) pela implementação do YOLO
- [OpenCV](https://opencv.org/) pelas capacidades de processamento de imagem