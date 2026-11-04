# N1_IA_VisaoComputacional
# Detecção de malária com CNN: 5 classes × 2 classes

Trabalho de **Inteligência Artificial** (EC E-954), Engenharia de Computação, FESA, turma EC09, Prof. Marcones Cleber.
Grupo: _(nomes e RA)_

Uma rede neural convolucional (CNN), construída camada por camada em **Keras**, classifica imagens de hemácias da **versão revisada** do conjunto de malária do NIH (IEEE DataPort). O trabalho tem três partes:
- **Parte A:** as 5 classes da versão revisada.
- **Parte B:** só *Parasitized* × *Uninfected*.
- **Parte C:** comparação entre as duas.

---

## Arquivos

| Arquivo | Descrição |
|---|---|
| `CNN_Malaria_IEEE_5x2.ipynb` | Notebook principal, comentado, para rodar no Google Colab |
| `CNN_Malaria_IEEE_5x2_executado.ipynb` | O mesmo notebook com as saídas de uma execução completa (CPU), só para consulta |
| `Apresentacao_CNN_Malaria_IEEE.pptx` | Apresentação de ~10 min, com roteiro de fala nas anotações dos slides |
| `resultados_malaria_ieee.zip` | Figuras e tabelas geradas pelo notebook (`figs/`) |

---

## Contexto

A malária é causada pelo *Plasmodium* e transmitida pelo mosquito *Anopheles*. As lâminas deste conjunto são de ***P. falciparum***. No sangue periférico dessa espécie aparecem principalmente **anéis** (trofozoítos jovens), muitas vezes colados na borda da hemácia, porque as formas maduras ficam presas nos capilares. Na coloração de Giemsa, o parasita aparece roxo sobre a hemácia rosa.

## Dados

- **Fonte:** [Malaria dataset (thin blood smear, updated) – IEEE DataPort](https://ieee-dataport.org/open-access/malaria-dataset-thin-blood-smear-updated), arquivo `malaria_thin_smear.zip`. O download exige conta IEEE gratuita.
- **Origem:** lâminas de 150 pacientes com *P. falciparum* e 50 saudáveis (Bangladesh). Esta é a versão revisada e reclassificada do conjunto original do NIH/Kaggle.

| Classe | Imagens | % | Descrição |
|---|---|---|---|
| Uninfected | 14.141 | 51,3 | hemácia sem parasita |
| Parasitized | 12.057 | 43,8 | hemácia com parasita |
| Bad segmentation | 571 | 2,1 | recorte ruim da célula |
| Unsure | 570 | 2,1 | o revisor não teve certeza se há parasita |
| Weird | 218 | 0,8 | aparência atípica |

**Principais achados da análise exploratória**
- **Desbalanceamento forte:** a maior classe tem 65 vezes mais imagens que a menor. Por isso usamos o **F1 macro** para escolher os modelos.
- **Comparação com o conjunto original do Kaggle**, imagem por imagem: a revisão moveu **448 imagens de *Parasitized* para *Uninfected***. As classes novas vieram quase todas de *Parasitized*.
- **Resolução:** mediana de 130×130 px. Todas as imagens foram redimensionadas para 64×64.
- **Duplicatas:** nenhuma (verificação pelo hash MD5).
- **Lâminas:** são 201. As 47 que só têm *Uninfected* são de pacientes saudáveis. Por isso a divisão dos dados é feita **por lâmina**.

---

## Como executar

1. Baixe `malaria_thin_smear.zip` no IEEE DataPort e coloque no Google Drive (`MyDrive/malaria_thin_smear.zip`).
2. Abra `CNN_Malaria_IEEE_5x2.ipynb` no Colab e escolha **GPU T4** (na CPU, a execução leva cerca de 2 horas).
3. Execute **Ambiente de execução → Executar tudo** e autorize o acesso ao Drive.
4. No final, o arquivo `resultados_malaria_ieee.zip` é baixado automaticamente.

**Reprodutibilidade:** a semente é fixa (`SEED = 42`). Na GPU podem aparecer diferenças pequenas entre execuções.

---

## Metodologia

1. **Pré-processamento:** RGB, 64×64, normalização para [0, 1] dentro do modelo.
2. **Divisão sem vazamento:** `StratifiedGroupKFold` por lâmina, estratificado pelas 5 classes. A Parte B usa a **mesma divisão**, sem as 3 classes novas.

   | Conjunto | Uninf. | Paras. | Bad seg. | Unsure | Weird | Total | Lâminas |
   |---|---|---|---|---|---|---|---|
   | Treino | 9.119 | 7.693 | 366 | 360 | 144 | 17.682 | 130 |
   | Validação | 2.192 | 1.879 | 96 | 94 | 31 | 4.292 | 31 |
   | Teste | 2.830 | 2.485 | 109 | 116 | 43 | 5.583 | 40 |

3. **CNN baseline:** `[Conv 3×3 + ReLU + MaxPool 2×2] × 3 (16, 32, 64 filtros) → Flatten → Dense 64 → Dense softmax (5 ou 2)`. São 286.117 parâmetros com 5 classes e 285.922 com 2. Treino com Adam (lr = 0,001), no máximo 15 épocas.
4. **Experimentos:** baseline + 7 experimentos, mudando **um fator por vez** a partir da melhor configuração: data augmentation, dropout, batch normalization, um bloco a mais, taxa de aprendizado, early stopping e pesos de classe. A mudança é mantida se o **F1 macro de validação** subir.
5. **Avaliação final:** teste usado **uma única vez** em cada parte, com matriz de confusão, métricas por classe e Grad-CAM.

---

## Resultados

> Valores da execução registrada em `CNN_Malaria_IEEE_5x2_executado.ipynb` (CPU). Uma execução no Colab com GPU pode variar levemente. A comparação com a MobileNetV2 pré-treinada não foi executada nessa versão.

### Experimentos (F1 macro na validação, %)

| Exp. | Fator | Parte A (5 classes) | Parte B (2 classes) |
|---|---|---|---|
| Baseline | — | 52,94 | 97,36 |
| E1 | Data augmentation | 50,97 | **97,83** ✓ |
| E2 | Dropout 0,5 | 46,43 | 97,80 |
| E3 | Batch Normalization | 49,71 | 97,31 |
| E4 | +1 bloco conv (128) | 49,63 | **97,97** ✓ |
| E5 | lr 1e-3 → 3e-4 | **55,29** ✓ | **98,07** ✓ |
| E6 | Early stopping | 51,89 | 97,90 |
| E7 | Pesos de classe | 40,78 | 97,95 |

✓ = mudança mantida.
- **Modelo final A:** baseline com lr 3e-4.
- **Modelo final B:** augmentation + 4º bloco + lr 3e-4 (228.706 parâmetros).

### Teste

| | Parte A (5 classes) | Parte B (2 classes) |
|---|---|---|
| Acurácia | 91,6% | **97,8%** |
| F1 macro | 45,7% | 97,8% |
| Sensibilidade (recall *Parasitized*) | 93,6% | **97,5%** |

**F1 por classe na Parte A:** Uninfected 95,2% · Parasitized 92,8% · Bad segmentation 20,7% · Unsure 0,0% · Weird 19,6%.

**Comparação nas mesmas 5.315 imagens P/U do teste.** As previsões do modelo de 5 classes foram agrupadas em "parasitada" e "não parasitada":

| | 5 classes | 2 classes |
|---|---|---|
| Acurácia | 96,0% | 97,8% |
| Sensibilidade | 93,6% | 97,5% |
| Especificidade | 98,2% | 98,1% |

### Análise de erros
- **Unsure e Weird** são previstas quase sempre como *Parasitized*. Essas imagens vieram de *Parasitized* e têm manchas roxas reais.
- **Pesos de classe (E7)** desestabilizaram o treino: o peso de um erro em *Weird* era cerca de 25, contra 0,4 em *Uninfected*.
- **Falsos positivos da Parte B:** vários vêm de lâminas de pacientes saudáveis. A mancha roxa fica fora da hemácia, provavelmente uma plaqueta ou resto de corante.
- **Grad-CAM:** nos acertos de *Parasitized*, a rede foca o anel do parasita.

---

## Conclusão

- Para triagem, o modelo de **2 classes** é melhor: tem 3,9 pontos percentuais a mais de sensibilidade.
- O modelo de 5 classes poderia sinalizar imagens duvidosas, mas ainda erra demais nas classes raras.

**Limitações**
- As classes novas são pequenas e definidas de forma subjetiva.
- A validação tem poucas imagens raras, então a escolha pelo F1 macro fica ruidosa.
- As imagens foram reduzidas para 64×64.
- Há uma única espécie e uma única origem.

**Melhorias possíveis**
- Oversampling ou *focal loss* para as classes raras.
- Validação cruzada por lâmina.
- Ajustar o limiar de decisão para priorizar a sensibilidade.
- Imagens maiores, com GPU.
- *Fine-tuning* de modelos pré-treinados.

---

## Uso de IA generativa

O grupo usou o Claude (Anthropic) como apoio. O log de uso, com as correções críticas, está na **seção 11 do notebook**. Todos os números e gráficos vêm da execução real do notebook.

## Referências
- Rajaraman S. et al. *Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images.* PeerJ 6:e4568, 2018.
- [IEEE DataPort – Malaria dataset (thin blood smear, updated)](https://ieee-dataport.org/open-access/malaria-dataset-thin-blood-smear-updated)
- [NIH/NLM – Malaria Screener: repositório de dados](https://lhncbc.nlm.nih.gov/LHC-downloads/mobile-app.html)
- [CDC – Laboratory diagnosis of malaria: *P. falciparum*](https://cdc.gov/dpdx/resources/pdf/benchAids/malaria/Pfalciparum_benchaidV2.pdf)
- Selvaraju R. R. et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017.
