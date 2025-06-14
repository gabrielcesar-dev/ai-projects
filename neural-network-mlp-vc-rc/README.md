# Explicação Detalhada: Rede Neural MLP e Conceitos Fundamentais

## Visão Geral da Implementação

A rede neural desenvolvida é uma **Multi-Layer Perceptron (MLP)** com arquitetura de **3 camadas**, organizada da seguinte forma:

```
Entrada (1 neurônio) → Camada Oculta (15 neurônios) → Saída (1 neurônio)
```

### Fluxo de Funcionamento:

1. **Inicialização:**  
Os pesos das conexões são definidos com valores aleatórios para evitar simetria entre os neurônios.

2. **Feedforward:**  
Os dados de entrada são propagados da camada de entrada até a camada de saída, gerando a predição.

3. **Cálculo do Erro:**  
A saída gerada pela rede é comparada com a saída esperada (valor alvo), calculando o erro.

4. **Backpropagation:**  
O erro é propagado de volta pelas camadas, ajustando os pesos com base nos gradientes calculados.

5. **Repetição:**  
Este processo é repetido por várias épocas até que o erro atinja um valor aceitável.

---

## Funcionamento Detalhado

### 1. Feedforward (Propagação para Frente)

**Matemática:**

```
Camada Oculta:    z₀ = X · W₀
                  a₀ = sigmoid(z₀)

Camada de Saída:  z₁ = a₀ · W₁  
                  a₁ = sigmoid(z₁)  ← Saída final
```

**Exemplo de Código:**

```python
def feedforward(self):
    # Camada oculta
    self.pot_ativ_0 = np.dot(self.entrada, self.pesos_0)  # z₀
    self.camada_0 = sigmoid(self.pot_ativ_0)              # a₀

    # Camada de saída
    self.pot_ativ_1 = np.dot(self.camada_0, self.pesos_1) # z₁
    self.camada_1 = sigmoid(self.pot_ativ_1)              # a₁

    return self.camada_1
```

---

### 2. Backpropagation (Retropropagação do Erro)

**Objetivo:**  
Ajustar os pesos da rede para minimizar o erro total.

**Aplicação da Regra da Cadeia:**

```
∂MSE/∂W₁ = ∂MSE/∂a₁ · ∂a₁/∂z₁ · ∂z₁/∂W₁

∂MSE/∂W₀ = ∂MSE/∂a₁ · ∂a₁/∂z₁ · ∂z₁/∂a₀ · ∂a₀/∂z₀ · ∂z₀/∂W₀
```

**Principais Derivadas Utilizadas:**

- ∂MSE/∂a₁ = 2(y - ŷ): Derivada da função de custo MSE.
- ∂a₁/∂z₁ = sigmoid'(z₁): Derivada da função de ativação.
- ∂z₁/∂W₁ = a₀: Saída da camada anterior.
- ∂z₁/∂a₀ = W₁: Pesos da camada de saída.

**Exemplo de Código:**

```python
def backprop(self):
    # Gradiente da camada de saída
    erro_saida = 2 * (self.y - self.saida)
    gradiente_saida = erro_saida * sigmoid_derivative(self.pot_ativ_1)
    d_pesos_1 = np.dot(self.camada_0.T, gradiente_saida)

    # Gradiente da camada oculta
    erro_oculta = np.dot(gradiente_saida, self.pesos_1.T)
    gradiente_oculta = erro_oculta * sigmoid_derivative(self.pot_ativ_0)
    d_pesos_0 = np.dot(self.entrada.T, gradiente_oculta)

    # Atualização dos pesos
    self.pesos_0 += d_pesos_0 * learning_rate
    self.pesos_1 += d_pesos_1 * learning_rate
```

---

### 3. Processo de Aprendizado

A cada época, os seguintes passos são executados:

```python
for epoca in range(500):
    saida = nn.feedforward()        # 1. Calcular predição
    erro = mse(y_real, saida)       # 2. Medir erro
    nn.backprop()                   # 3. Ajustar pesos
```

---

## Por que calcular a derivada parcial?

O cálculo da derivada parcial ∂E/∂wᵢ é essencial para o processo de ajuste de pesos na rede neural, por três motivos principais:

1. **Sensibilidade local:**  
Indica quanto o erro total E varia ao modificar apenas o peso wᵢ, mantendo os demais constantes.

2. **Direção de ajuste:**  
Se ∂E/∂wᵢ for positiva, um aumento em wᵢ aumentará o erro, então o peso deve ser reduzido. Se for negativa, o peso deve ser aumentado.

3. **Intensidade da atualização:**  
A magnitude da derivada determina o tamanho do ajuste necessário: quanto maior a derivada, maior a atualização no peso.

### Processo de Otimização por Gradiente

O treinamento da rede neural segue o seguinte ciclo:

1. **Cálculo do gradiente:**  
∇E = [∂E/∂w₁, ∂E/∂w₂, ..., ∂E/∂wₙ]

2. **Atualização dos pesos:**  
wᵢ ← wᵢ - α · ∂E/∂wᵢ  
Onde α é a taxa de aprendizado.

3. **Iteração até a convergência:**  
O processo é repetido até que o erro seja minimizado.

Esse ciclo é executado eficientemente pelo algoritmo de backpropagation, que aplica a regra da cadeia para propagar os gradientes da camada de saída até as camadas iniciais.

---

## Por que usar a função Sigmoid? Por que a ReLU é mais usada?

### Função Sigmoid

**Definição matemática:**  
σ(x) = 1 / (1 + e^(-x))

**Derivada:**  
σ'(x) = σ(x) · (1 - σ(x))

**Vantagens da função Sigmoid:**

- Limita a saída entre 0 e 1, permitindo interpretação probabilística.
- Função contínua e diferenciável em todos os pontos.
- Introduz não linearidade, essencial para o aprendizado de padrões complexos.
- Controla a amplitude da saída, evitando valores extremos.

**Desvantagens da função Sigmoid:**

- Problema de vanishing gradient: em entradas muito altas ou muito baixas, a derivada se aproxima de zero, prejudicando o aprendizado em redes profundas.
- Alto custo computacional, devido à presença de operações exponenciais.
- Saturação: em valores extremos de entrada, a saída se aproxima dos limites da função (0 ou 1), reduzindo a sensibilidade do gradiente.
- Não centrada em zero: a saída sempre positiva pode prejudicar a eficiência na atualização dos pesos.

---

### Função ReLU (Rectified Linear Unit)

**Definição matemática:**  
ReLU(x) = max(0, x)

**Derivada:**  
ReLU'(x) = 1 se x > 0, caso contrário 0

**Principais vantagens da ReLU:**

1. **Redução do vanishing gradient:**  
Para x > 0, o gradiente é constante e igual a 1, permitindo uma propagação eficiente dos gradientes mesmo em redes profundas.

2. **Eficiência computacional:**  
Sua implementação é simples e rápida, exigindo apenas uma comparação por operação.

3. **Esparsidade natural:**  
Em média, cerca de metade dos neurônios permanece com saída zero, tornando a rede mais eficiente, com menos consumo de recursos e menor risco de overfitting.

4. **Convergência mais rápida:**  
A estabilidade e previsibilidade dos gradientes proporcionam um treinamento mais rápido, com menor número de épocas para atingir uma boa precisão.

---
