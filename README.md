# MOD17 OPTIMIZATION AND PREDICTION
Código utilizado no artigo "ENHANCING MODIS SATELLITE GPP PRODUCT ESTIMATION IN THE AMAZON REGION THROUGH PARAMETER OPTIMIZATION AND TIME SERIES DEEP LEARNING MODELING" submetido na revista 'GIScience & Remote Sensing' e aguardando revisão.

## Motivação
- A Tabela BPLUT possui valores estáticos, que não sofrem mudanças ao longo do tempo e do espaço dentro dos Biomas. Isso gera incerteza nos cálculos de GPP do algoritmo do MOD17.
- Para compreender a capacidade de absorção de carbono do bioma amazônico, uma métrica importante para ser observada é o GPP (Gross Primary Productivity), representando a quantidade total de carbono que as plantas captam da amosfera por meio da fotossíntese, antes de qualquer respiração ou perda de carbono.

## Objetivos
- Atualizar a tabela BPLUT mensalmente para o Bioma Amazônico, resultando em parâmeros mais precisos para o cálculo do GPP.
- Treinar uma rede neural para predição de GPP para a Amazônia, utilizando valores mais precisos resultantes dos cálculos com os parãmetros mensais da BPLUT.

## Resumo
O melhor monitoramento do fluxo de carbono na atmosfera tem uma profunda importância na compreensão do impacto da captura de carbono pela Floresta Amazônica. Nesse sentido, o presente artigo busca prover um modelo de previsão de série temporal de Gross Primary Productivity (GPP) para a região. Este estudo foca em utilizar inteligência artificial, com aplicação do algoritmo genético em intervalos mensais para aprimorar os parâmetros do algoritmo "MOD17" que estima o GPP com base em dados de sensoriamento remoto, usando dados de três torres de fluxo na região (BR-Sa1, PE-QRF e BR-Cax) como referência. A otimização do algoritmo genético trouxe melhoras por mês na Raiz do Erro Médio Quadrático (RSME) de até 36.85% em BR-Sa1, de até 41.47% em PE-QRF, e de até 54.77% para BR-Cax. Em seguida, foi escolhido e treinado um modelo de deep learning para aprender os padrões das séries temporais pós-otimização. Nos testes, após a seleção do InceptionTimePlus como o melhor modelo, este apresentou uma correlação de 0.82 e um RMSE de 1.93 nos dados de GPP otimizados.
