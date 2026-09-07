# 🍃 Classificação de Folhas utilizando Visão Computacional e Machine Learning

Projeto desenvolvido para **classificação de espécies de folhas** utilizando técnicas de **Processamento Digital de Imagens, Extração de Características e Aprendizado de Máquina**.

O projeto realiza o processamento das imagens das folhas, extrai diferentes descritores de **textura e forma** e utiliza essas características para treinar e avaliar diferentes algoritmos de classificação.

---

## 🎯 Objetivo

O objetivo do projeto é investigar a capacidade de diferentes técnicas de **extração de características** e diferentes **classificadores de Machine Learning** na identificação de espécies de folhas.

A pipeline pode ser resumida da seguinte forma:

```text
                    Imagens das folhas
                           │
                           ▼
                  Pré-processamento
                           │
                           ▼
                    Segmentação
                    (Otsu + Blur)
                           │
                           ▼
                Extração de características
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
           GLCM           LBP       Momentos de Hu
             │             │             │
             └─────────────┼─────────────┘
                           │
                           ▼
                  Características de
                         forma
                           │
                           ▼
                    Vetor de atributos
                           │
                           ▼
                ┌─────────────────────┐
                │    Classificadores  │
                ├─────────────────────┤
                │ Gaussian Naive Bayes│
                │ KNN                 │
                │ MLP                 │
                │ SVM                 │
                └─────────────────────┘
                           │
                           ▼
                    Avaliação dos
                       modelos
```

---

# 🔬 Metodologia

## 1. Pré-processamento

As imagens são carregadas em escala de cinza utilizando **OpenCV**.

Inicialmente é aplicado um filtro Gaussiano:

```python
blur = cv2.GaussianBlur(img, (5, 5), 0)
```

Em seguida, é realizada a segmentação da folha utilizando **limiarização de Otsu**:

```python
_, binario = cv2.threshold(
    blur,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)
```

O resultado é utilizado como máscara para separar a folha do fundo.

---

# 🧩 Extração de características

Após a segmentação, são extraídas diferentes características das imagens.

O projeto trabalha principalmente com quatro grupos de descritores:

* **GLCM** — características de textura;
* **LBP** — padrões locais de textura;
* **Momentos de Hu** — características invariantes de forma;
* **Características geométricas** — propriedades estruturais da folha.

A implementação da extração é realizada no arquivo `Analise.py`.

---

## 🧱 GLCM — Gray-Level Co-occurrence Matrix

A **GLCM (Gray-Level Co-occurrence Matrix)** é utilizada para representar a distribuição espacial dos níveis de cinza de uma imagem.

Neste projeto são extraídas as seguintes propriedades:

* Dissimilaridade;
* Correlação;
* Homogeneidade;
* ASM;
* Energia.

A implementação utiliza `graycomatrix()` e `graycoprops()` da biblioteca `scikit-image`.

Essas características ajudam a representar a **textura da superfície da folha**.

---

## 🔳 LBP — Local Binary Pattern

O **LBP (Local Binary Pattern)** é utilizado para representar padrões locais de textura.

O projeto utiliza:

```python
local_binary_pattern(
    img_segmentada,
    P=8,
    R=1,
    method="default"
)
```

A partir do histograma LBP são calculadas características estatísticas como:

* Média;
* Variância;
* Desvio padrão;
* Suavidade;
* Terceiro momento;
* Entropia.

Essas características são posteriormente utilizadas pelos classificadores.

---

# 📐 Momentos de Hu

Os **sete Momentos de Hu** são utilizados para representar propriedades relacionadas à forma do objeto.

O projeto calcula os momentos utilizando:

```python
momentos = cv2.moments(binario)
hu = cv2.HuMoments(momentos).flatten()
```

Em seguida, os valores são transformados para uma escala logarítmica para facilitar sua utilização como atributos:

```python
hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-15)
```

São utilizados os sete momentos:

```text
Hu_1
Hu_2
Hu_3
Hu_4
Hu_5
Hu_6
Hu_7
```

---

# 📏 Características de forma

Além dos descritores de textura, o projeto calcula características geométricas da folha.

São utilizadas:

| Característica           | Descrição                                                 |
| ------------------------ | --------------------------------------------------------- |
| **Excentricidade**       | Mede o quanto a forma se aproxima de uma elipse alongada  |
| **Razão de aspecto**     | Relação entre altura e largura                            |
| **Alongamento**          | Mede características relacionadas ao alongamento da folha |
| **Solidez**              | Relação entre a área da região e seu casco convexo        |
| **Fator isoperimétrico** | Relaciona área e perímetro                                |
| **Convexidade**          | Relação entre o contorno e seu casco convexo              |

Essas características são calculadas a partir da região segmentada da folha.

---

# 📊 Construção do dataset

Depois da extração, todas as características são organizadas em um `DataFrame` do Pandas.

O resultado é salvo no arquivo:

```text
Atributos.csv
```

O arquivo contém:

```text
id
y
glcm_*
lbp_*
Hu_*
forma_*
```

A coluna `y` representa a classe da folha.

---

# 🤖 Classificação

O projeto compara quatro algoritmos de Machine Learning.

## 1. Gaussian Naive Bayes

Implementado em:

```text
gaussiannb.py
```

O modelo utilizado é:

```python
GaussianNB()
```

---

## 2. K-Nearest Neighbors — KNN

Implementado em:

```text
knn.py
```

O modelo utiliza:

```python
KNeighborsClassifier(n_neighbors=5)
```

Ou seja, são considerados os **5 vizinhos mais próximos** para realizar a classificação.

---

## 3. Multi-Layer Perceptron — MLP

Implementado em:

```text
mlp.py
```

O projeto utiliza o `MLPClassifier` do Scikit-Learn:

```python
MLPClassifier(
    random_state=42,
    max_iter=2500
)
```

O MLP permite avaliar uma abordagem baseada em **rede neural artificial** sobre as características extraídas das folhas.

---

## 4. Support Vector Machine — SVM

Implementado em:

```text
svm.py
```

O classificador utilizado é:

```python
SVC(
    kernel="rbf",
    C=1.0,
    random_state=42
)
```

É utilizado o kernel **RBF (Radial Basis Function)**.

---

# 🧪 Combinações de características

Uma das partes importantes do projeto é a comparação entre diferentes conjuntos de atributos.

São avaliadas três combinações:

### GLCM + Forma

```text
GLCM
  +
Características de forma
```

### LBP + Forma

```text
LBP
  +
Características de forma
```

### Momentos de Hu + Forma

```text
Momentos de Hu
       +
Características de forma
```

Essas combinações são utilizadas pelos quatro classificadores.

---

# ⚙️ Pré-processamento dos atributos

Antes do treinamento dos modelos, os atributos passam por **padronização utilizando `StandardScaler`**.

O processo é:

```python
scaler = StandardScaler()

X_scaler_train = scaler.fit_transform(X_train)
X_scaler_test = scaler.transform(X_test)
```

O `StandardScaler` é ajustado somente com os dados de treinamento e posteriormente aplicado aos dados de teste, evitando que informações do conjunto de teste sejam utilizadas durante o ajuste do escalonamento.

---

# 🔄 Validação cruzada

Para avaliar os modelos é utilizada **Stratified K-Fold Cross Validation**.

A configuração utilizada é:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Isso significa que o dataset é dividido em **5 folds**, mantendo a proporção das classes durante a divisão.

O processo pode ser representado como:

```text
              Dataset
                 │
        ┌────────┼────────┐
        ▼        ▼        ▼
      Fold 1   Fold 2   Fold 3 ...
        │
        ▼
   Treinamento
        +
      Teste
        │
        ▼
     Métricas
        │
        ▼
      Média
        ±
     Desvio
```

---

# 📈 Métricas

Os modelos são avaliados utilizando diferentes métricas de classificação.

### Acurácia

Mede a proporção geral de classificações corretas:

```text
Acurácia =
classificações corretas
──────────────────────
total de amostras
```

### Sensibilidade

Também chamada de **Recall**, representa a capacidade do modelo de identificar corretamente as amostras pertencentes às classes.

### Especificidade

Mede a capacidade do classificador de identificar corretamente as amostras que não pertencem a determinada classe.

### F1-Score

Combina precisão e sensibilidade em uma única métrica:

```text
F1 = 2 × (Precisão × Recall)
          ────────────────
           Precisão + Recall
```

Além das métricas numéricas, o projeto também calcula a **matriz de confusão** para analisar os erros de classificação.

---

# 📁 Estrutura do projeto

```text
Trabalho_de_classificacao_de_folhas/
│
├── Analise.py
│
├── gaussiannb.py
├── knn.py
├── mlp.py
├── svm.py
│
├── Relatório_da_atividade_de_classificação.pdf
│
└── README.md
```

### `Analise.py`

Responsável pelo processamento das imagens e pela **extração das características**.

O script gera o arquivo:

```text
Atributos.csv
```

### `gaussiannb.py`

Implementação do classificador **Gaussian Naive Bayes**.

### `knn.py`

Implementação do classificador **K-Nearest Neighbors**.

### `mlp.py`

Implementação do classificador **Multi-Layer Perceptron**.

### `svm.py`

Implementação do classificador **Support Vector Machine** com kernel RBF.

### `Relatório_da_atividade_de_classificação.pdf`

Relatório acadêmico relacionado ao desenvolvimento e análise do projeto. O arquivo está disponível diretamente no repositório.

---

# 🛠️ Tecnologias utilizadas

| Tecnologia          | Utilização                              |
| ------------------- | --------------------------------------- |
| 🐍 **Python**       | Linguagem principal                     |
| 🖼️ **OpenCV**      | Processamento e segmentação das imagens |
| 📊 **Pandas**       | Manipulação dos dados                   |
| 🔢 **NumPy**        | Operações matemáticas                   |
| 🔬 **Scikit-Image** | GLCM, LBP e análise de regiões          |
| 🤖 **Scikit-Learn** | Machine Learning e métricas             |
| 📄 **CSV**          | Armazenamento dos atributos             |

---

# 📦 Bibliotecas

As principais bibliotecas utilizadas no projeto são:

```text
numpy
pandas
opencv-python
scikit-image
scikit-learn
```

Instalação:

```bash
pip install numpy pandas opencv-python scikit-image scikit-learn
```

---

# 🚀 Como executar

## 1. Clone o repositório

```bash
git clone https://github.com/Cvtr-C/Trabalho_de_classificacao_de_folhas.git
```

## 2. Entre no diretório

```bash
cd Trabalho_de_classificacao_de_folhas
```

## 3. Instale as dependências

```bash
pip install numpy pandas opencv-python scikit-image scikit-learn
```

## 4. Extraia as características

Execute:

```bash
python Analise.py
```

O programa irá processar as imagens e gerar:

```text
Atributos.csv
```

## 5. Execute os classificadores

### Gaussian Naive Bayes

```bash
python gaussiannb.py
```

### KNN

```bash
python knn.py
```

### MLP

```bash
python mlp.py
```

### SVM

```bash
python svm.py
```

---

# 🧠 Pipeline completa

O funcionamento geral do projeto pode ser resumido em:

```text
┌──────────────────────┐
│    Imagens de folhas │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Conversão para       │
│ escala de cinza      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Filtro Gaussiano     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Segmentação por Otsu │
└──────────┬───────────┘
           │
           ▼
┌────────────────────────────┐
│ Extração de características│
├────────────────────────────┤
│ • GLCM                     │
│ • LBP                      │
│ • Momentos de Hu           │
│ • Forma                    │
└──────────┬─────────────────┘
           │
           ▼
┌──────────────────────┐
│     Atributos.csv    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ StandardScaler       │
└──────────┬───────────┘
           │
           ▼
┌───────────────────────────────┐
│      Stratified K-Fold        │
│          5 folds              │
└──────────┬────────────────────┘
           │
           ▼
┌───────────────────────────────┐
│          Classificação        │
├───────────────────────────────┤
│ Gaussian Naive Bayes          │
│ KNN                           │
│ MLP                           │
│ SVM                           │
└──────────┬────────────────────┘
           │
           ▼
┌───────────────────────────────┐
│          Avaliação            │
├───────────────────────────────┤
│ Acurácia                      │
│ Sensibilidade                 │
│ Especificidade                │
│ F1-Score                      │
│ Matriz de Confusão            │
└───────────────────────────────┘
```

---

# 📌 Principais conceitos abordados

Este projeto reúne conceitos de diferentes áreas da Computação:

### 🖼️ Processamento Digital de Imagens

* Conversão para escala de cinza;
* Filtro Gaussiano;
* Limiarização de Otsu;
* Segmentação;
* Contornos;
* Transformada de distância.

### 🔬 Extração de características

* GLCM;
* LBP;
* Momentos de Hu;
* Excentricidade;
* Solidez;
* Convexidade;
* Razão de aspecto;
* Fator isoperimétrico.

### 🤖 Machine Learning

* Gaussian Naive Bayes;
* KNN;
* MLP;
* SVM;
* Normalização/padronização;
* Validação cruzada estratificada.

### 📊 Avaliação

* Acurácia;
* Sensibilidade;
* Especificidade;
* F1-Score;
* Matriz de confusão;
* Média e desvio padrão.

---

# 🎓 Contexto acadêmico

Este projeto foi desenvolvido como uma atividade acadêmica relacionada à **classificação de imagens**, aplicando técnicas clássicas de **Visão Computacional e Aprendizado de Máquina**.

A abordagem utilizada é baseada em **engenharia de características (feature engineering)**: em vez de fornecer diretamente as imagens aos classificadores, são extraídas características matemáticas que representam propriedades relevantes das folhas.

Isso permite estudar separadamente:

```text
Imagem
  ↓
Processamento
  ↓
Características
  ↓
Classificador
  ↓
Predição
```

---

# 👨‍💻 Autor

**Cvtr-C**

GitHub:

https://github.com/Cvtr-C

---

## 📄 Licença

Projeto desenvolvido para fins **acadêmicos e educacionais**.
