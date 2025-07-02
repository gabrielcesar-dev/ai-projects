# Projeto: Clusterização com DBSCAN usando Python e Scikit-learn

## Introdução

Este projeto tem como objetivo apresentar o funcionamento do algoritmo de clusterização **DBSCAN** (*Density-Based Spatial Clustering of Applications with Noise*).O DBSCAN é uma técnica poderosa de **aprendizado não supervisionado**, capaz de formar agrupamentos com base na **densidade dos dados** e identificar automaticamente **outliers (ruídos)** — dados que não pertencem a nenhum cluster.

---

## O que é Clusterização?

**Clusterização** é o processo de agrupar elementos semelhantes com base em características compartilhadas. Ela é uma forma de **aprendizado não supervisionado**, pois os dados não possuem rótulos prévios.

> “Cluster analysis or clustering is the task of grouping a set of objects in such a way that objects in the same group (called a cluster) are more similar to each other than to those in other groups.”  
> — *Wikipedia (2004)*

---

## O que é o DBSCAN?

**DBSCAN** é um algoritmo de clusterização baseado em densidade. Ele identifica **áreas densas de dados** e forma agrupamentos ao redor delas. Além disso, lida muito bem com **dados ruidosos**, marcando pontos isolados como **outliers**, o que o torna ideal para análises com dados reais.

### Principais vantagens:
- Não requer o número de clusters como entrada.
- Detecta clusters com **formas arbitrárias**.
- Lida bem com **ruído e outliers**.

---

## Conceitos Fundamentais

Antes de entender como o DBSCAN funciona, é importante conhecer seus principais conceitos:

### Core Point (Ponto Central)
Um ponto é considerado central se houver pelo menos `min_samples` pontos dentro de uma distância ε (epsilon) dele, incluindo ele mesmo.

### Directly Reachable (Diretamente Acessível)
Um ponto **q** é diretamente acessível a partir de um **core point p**, se estiver dentro do raio ε de **p**.

### Reachable (Acessível)
Um ponto é acessível se houver um caminho de pontos centrais até ele, mesmo que ele próprio não seja um ponto central.

### Outlier (Ruído)
Pontos que não são acessíveis a partir de nenhum ponto central são considerados **ruído**.

---

## Parâmetros do DBSCAN

- **`eps (ε)`**: Raio da vizinhança. Define o quão próximos os pontos precisam estar para serem considerados vizinhos.
- **`min_samples`**: Número mínimo de pontos que devem estar dentro do raio `ε` para que um ponto seja considerado central.

---

## Como o DBSCAN Funciona (Passo a Passo)

O funcionamento do DBSCAN pode ser descrito em etapas:

1. **Definir `eps` e `min_samples`**: Esses dois parâmetros guiam todo o algoritmo.

2. **Selecionar um ponto não visitado aleatoriamente**:
   - Se houver pelo menos `min_samples` vizinhos dentro do raio `ε`, o ponto é um **core point**, e um novo **cluster é iniciado**.
   - Caso contrário, o ponto é marcado como **ruído** (mas isso pode mudar depois).

3. **Expandir o cluster**:
   - Para cada ponto dentro da vizinhança do core point, verifique seus próprios vizinhos.
   - Se também for um core point, adicione seus vizinhos ao cluster.
   - Repita o processo até que não haja mais pontos a adicionar.

4. **Repetir**:
   - Continue escolhendo pontos não visitados e repita os passos anteriores até que todos tenham sido processados.

5. **Resultado final**:
   - Todos os pontos são rotulados como parte de um cluster ou como **ruído**.

---

## Escolhendo o valor ideal de `eps`

Uma técnica comum para definir um bom valor de `ε` é o **gráfico de k-distância (Elbow Method)**:

1. Defina um valor para `min_samples` (ex: 4 para dados 2D).
2. Calcule a distância de cada ponto ao seu k-ésimo vizinho mais próximo (k = `min_samples` - 1).
3. Ordene e plote as distâncias.
4. O "cotovelo" do gráfico indica um bom valor de `ε`.