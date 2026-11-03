# N1_IA_VisaoComputacional
# Detecção de malária em células com uma CNN

Trabalho de **Redes Neurais Convolucionais**, Engenharia de Computação (FESA).
Grupo:  Adriana Kaori Kakazu
        Beatriz Buglio dos Santos
        Lucas Oliveira Silva
        Vitoria Kaori Kuriyama

Uma rede neural convolucional (CNN), construída camada por camada em **Keras**, classifica a imagem de uma hemácia como **Parasitized** (infectada pelo *Plasmodium*) ou **Uninfected** (não infectada).

---

## Contexto: malária

A malária é causada pelo protozoário *Plasmodium* e transmitida pela fêmea do mosquito *Anopheles*. Dentro da hemácia o parasita passa pelas fases de **anel** (trofozoíto jovem), **trofozoíto maduro** e **esquizonte**, até a hemácia romper e liberar novos merozoítos. Alguns parasitas viram **gametócitos**.

As lâminas deste conjunto são de ***P. falciparum***. Nessa espécie, as hemácias com formas maduras ficam presas nos capilares, então no sangue periférico aparecem principalmente os **anéis**, muitas vezes colados na borda da célula (forma *accolé*). Na coloração de Giemsa, a cromatina do parasita fica roxo-avermelhada sobre a hemácia rosa. Por isso as imagens são usadas em cores (RGB).

---

## Conjunto de dados

- **Fonte:** [Cell Images for Detecting Malaria – Kaggle](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) (originalmente do NIH/NLM)
- **Origem:** lâminas de esfregaço fino coradas com Giemsa, de 150 pacientes com *P. falciparum* e 50 pessoas saudáveis, coletadas em Bangladesh e fotografadas com um celular acoplado ao microscópio
- **Tamanho:** 27.558 imagens de células isoladas, 13.779 por classe

**Principais achados da análise exploratória**
- As classes estão perfeitamente balanceadas (50% / 50%).
- A resolução é variável: mediana de 130×130 px, de 46×40 até 394×385 px. Todas as imagens foram redimensionadas para **64×64**.
- Não há duplicatas exatas (verificação pelo hash MD5).
- São **201 lâminas**. As 50 que só têm células *Uninfected* são as dos pacientes saudáveis. Por isso a divisão dos dados é feita **por lâmina**, para evitar vazamento.

---

## Como executar

1. Abra `CNN_Malaria_Grupo.ipynb` no [Google Colab](https://colab.research.google.com/).
2. Escolha **Ambiente de execução → Alterar o tipo de ambiente → GPU T4** (na CPU também funciona, mas leva cerca de 1h20).
3. Execute **Ambiente de execução → Executar tudo**.
4. O dataset é baixado automaticamente pelo `kagglehub`. Se o download falhar, o notebook tenta o link do NIH. Também é possível usar uma pasta própria em `DATA_DIR_MANUAL`.
5. No final, o arquivo `resultados_malaria.zip` com todas as figuras e tabelas é baixado.

**Dependências:** já vêm instaladas no Colab: `tensorflow`/`keras`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `Pillow`, `tqdm` e `kagglehub`.

**Reprodutibilidade:** a semente é fixa (`SEED = 42`). Na GPU podem aparecer diferenças muito pequenas entre execuções.

---

## Metodologia

1. **Pré-processamento:** conversão para RGB, redimensionamento para 64×64 e normalização para [0, 1] dentro do modelo (camada `Rescaling`).
2. **Divisão sem vazamento:** `StratifiedGroupKFold` por lâmina. Nenhuma lâmina aparece em dois conjuntos.

   | Conjunto | Imagens | % | Lâminas | % Parasitized |
   |---|---|---|---|---|
   | Treino | 17.597 | 63,9 | 128 | 50,2 |
   | Validação | 4.479 | 16,3 | 33 | 49,3 |
   | Teste | 5.482 | 19,9 | 40 | 50,0 |

3. **CNN baseline** (285.857 parâmetros):
   `Entrada 64×64×3 → [Conv 3×3 + ReLU + MaxPool 2×2] × 3 (16, 32, 64 filtros) → Flatten → Dense 64 → Dense 1 (sigmoid)`
   Treinada com o otimizador Adam (lr = 0,001) e perda `binary_crossentropy`, por no máximo 20 épocas.
4. **Experimentos:** a cada experimento muda **um fator por vez**, partindo da melhor configuração anterior. A mudança é mantida se a acurácia de validação subir.
5. **Avaliação final:** o conjunto de teste é usado **uma única vez**, só com o modelo escolhido.
6. **Explicabilidade:** Grad-CAM na última camada convolucional, aplicado a acertos e erros.
7. **Comparação:** MobileNetV2 pré-treinada e congelada, usada **apenas como referência**.

---

## Resultados

> Os valores abaixo vêm da execução registrada em `CNN_Malaria_Grupo_executado.ipynb` (CPU). Uma execução no Colab com GPU pode variar levemente. A comparação com a MobileNetV2 não foi executada nessa versão (`RODAR_TRANSFER = False`).

### Experimentos (validação)

| Exp. | Fator alterado | Acc val (%) | Gap treino−val (p.p.) | Resultado |
|---|---|---|---|---|
| Baseline | — | 95,09 | 4,77 | referência |
| E1 | Data augmentation (flip, rotação, zoom) | 96,07 | 0,42 | mantido |
| E2 | Dropout 0,5 | 95,98 | 0,44 | descartado |
| E3 | Batch Normalization | 95,65 | 0,70 | descartado |
| E4 | +1 bloco convolucional (128 filtros) | 96,23 | 0,27 | mantido |
| E5 | Taxa de aprendizado 1e-3 → 3e-4 | 95,51 | 0,37 | descartado |
| E6 | Early stopping (paciência 4) | 96,72 | −0,21 | mantido* |

\* O early stopping não chegou a interromper o treino. O ganho veio de `restore_best_weights`, que restaurou os pesos da melhor época (19).

**Modelo final:** baseline + data augmentation + 4º bloco + early stopping, com 228.641 parâmetros.

### Teste (modelo final)

| Classe | Precisão | Revocação | F1-score | Acurácia da classe |
|---|---|---|---|---|
| Parasitized | 96,21% | 95,33% | 95,77% | 95,33% |
| Uninfected | 95,37% | 96,24% | 95,81% | 96,24% |

- **Acurácia geral:** 95,79%
- **AUC-ROC:** 0,990
- **Matriz de confusão:** VN = 2.639 · FP = 103 · FN = 128 · VP = 2.612

### Análise de erros e Grad-CAM

- Nos acertos, o Grad-CAM se concentra sobre a mancha roxa do parasita, inclusive quando ela está na borda da célula.
- Vários **falsos positivos** muito confiantes mostram uma mancha roxa visível, e vários **falsos negativos** não mostram parasita. Isso sugere **rótulos errados** no conjunto, o que é coerente com a [revisão publicada do dataset](https://ieee-dataport.org/open-access/malaria-dataset-thin-blood-smear-updated). É uma hipótese.
- Os erros se concentram em algumas lâminas, com até 24% de erro numa lâmina contra 4,2% no geral.

---

## Limitações e melhorias

**Limitações**
- As células já vêm recortadas, não a lâmina inteira.
- Há uma única espécie e uma única origem.
- 50 lâminas só têm células de pacientes saudáveis.
- Pode haver ruído nos rótulos.
- As imagens são reduzidas para 64×64, e o limiar de decisão é fixo em 0,5.
- O Grad-CAM tem baixa resolução (mapa de 4×4).

**Melhorias possíveis**
- Validação cruzada por lâmina.
- Ajustar o limiar para priorizar a sensibilidade.
- Normalização de coloração.
- Imagens de 128×128 com GPU.
- *Fine-tuning* de modelos pré-treinados.
- Usar a versão revisada do conjunto.

O modelo deve **apoiar** o microscopista na triagem, não substituí-lo.

---

## Uso de IA generativa

O grupo usou o Claude (Anthropic) como apoio. O log com os prompts, as decisões do grupo e as correções críticas está na **seção 11 do notebook**. Todos os números e gráficos vêm da execução real do notebook.

---

## Referências

- Rajaraman S. et al. *Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images.* PeerJ, 6:e4568, 2018.
- [NIH/NLM – Malaria Screener: repositório de dados](https://lhncbc.nlm.nih.gov/LHC-downloads/mobile-app.html)
- [IEEE DataPort – Malaria dataset (thin blood smear, updated)](https://ieee-dataport.org/open-access/malaria-dataset-thin-blood-smear-updated)
- [CDC – Laboratory diagnosis of malaria: *Plasmodium falciparum*](https://cdc.gov/dpdx/resources/pdf/benchAids/malaria/Pfalciparum_benchaidV2.pdf)
- Selvaraju R. R. et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV, 2017.
