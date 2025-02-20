<img align= center src="https://user-images.githubusercontent.com/83380670/142748354-a649cd86-4e88-47a5-85fb-eda577893e22.png" width="100%" height="70%" />



# Análise Preditiva - Qualidade da Água
A  água é indispensável para vida humana e possui uma presença signifiativa no cotidiano das pessoas. De acordo com sua utilização, existe um conjunto de critérios e normas para a qualidade da água.
A água contém componentes, os quais provêm do próprio ambiente natural ou foram introduzidos a partir de atividades humanas. Sendo usada para diversos fins, como consumo humano, lazer, irrigação, entre outro e sua qualidade pode ser representada através de parâmetros que traduzem as suas principais características físicas, químicas e biológicas.
Verificar e acompanhar a qualidade da água é necessário para destinar seu uso.



## Objetivo
O objetivo do projeto é criar um modelo de predição da qualidade da água.

## Dataset

<h4>1. Qualidade da água </h4>

**Fonte:** Kaggle <br>
**Disponível em:** <https://www.kaggle.com/adityakadiwal/water-potability>



## Análise dos Dados:
Os dados foram analisados em Python com o auxílio do Jupyter Notebook. 
Na etapa de tratamento foi possível fazer a limpeza e transformação de variáveis afim de normalização dos dados. 
Foram aplicados 15 modelos de classificação 


**Modelos**
1. LogisticRegression
2. SVC
3. KNeighborsClassifier
4. DecisionTreeClassifier
5. GaussianNB
6. SGDClassifier
7. Perceptron
8. NearestCentroid
9. RidgeClassifier
10. NuSVC
11. BernoulliNB
12. RandomForestClassifier
13. AdaBoostClassifier
14. GradientBoostingClassifier
15. PassiveAggressiveClassifier

## Resultados
 
### Análise Exploratória

Inicialmente foi feito um estudo dos dados. Com o auxílio do gráfico Boxplot nota-se que todos os parâmetros possuim outliers superiores e inferiores.
Outliers são valores que destoam muito dos demais considerando para isso estar fora de um intervalo de distância interquartílica. Um tratamento adequado aos outliers é fundamental para que eles não representem desvios no modelo de regressão e classificação. Nesse projeto, afim de estudo, os outliers não serão tratados e nem retirados do conjunto de dados.

<img align= center src="https://user-images.githubusercontent.com/83380670/143305444-ecfe557a-1f98-41bb-b740-c6c9db9ac16a.png" width="70%" height="70%" />

Não há correlação linear entre os parâmetros estudados, fato que pode ser confirmado através da vizualização dos gráficos de dispersão entre os parâmentros e os valores da correlação de Spearman presentes na matriz de correlação.

<img align= center src="https://user-images.githubusercontent.com/83380670/143308198-7b30899a-57e7-4024-96c6-387b58db7a4c.png" width="70%" height="70%" />



<img align= center src="https://user-images.githubusercontent.com/83380670/143306995-ecc1d143-8acf-4a77-9d56-50b104815913.png" width="100%" height="70%" />


### Parâmetros

Potencial Hidrogeniônico (pH) é uma medida logarítima que varia de 0 a 14, utilizada para especificar a acidez ou basicidade de uma solução, se uma água é ácida (pH inferior a 7), neutra (pH igual a 7) ou alcalina (pH maior do que 7);
É um parâmetro que não tem risco sanitário associado diretamente à sua medida. Faixa recomendada de pH na água é de 6,0 a 9,5.



<img align= center src="https://user-images.githubusercontent.com/83380670/142760205-682658aa-b58f-4b34-b5e4-e4cc54d6d769.png" width="70%" height="70%" />

O parâmetro Dureza da água representa a presença de sais de metais alcalino terrosos, principalmente cátions de cálcio e magnésio.
Segundo a OMS (Organização Mundial de Saúde), uma água é designada por água muito dura
quando apresenta uma concentração em carbonato de cálcio superior a 180 mg/L;
dura com concentração entre 120 e 180 mg/L, moderadamente dura entre 60-120
mg/L e macia quando os teores em carbonato de cálcio são <60 mg/L.

<img align= center src="https://user-images.githubusercontent.com/83380670/142760232-8bc542a4-b678-4555-8653-e6ef03e63c1a.png" width="70%" height="70%" />

Sólidos Totais Dissolvidos (STD) é a medida da quantidade de material dissolvido na água. Composto por sais inorgânicos como cálcio, magnésio, cloretos, sulfatos, bicarbonatos, etc, junto com muitos outros compostos inorgânicos que se dissolvem facilmente na água.
É uma importante métrica de qualidade da água que se refere a todas as substâncias filtráveis orgânicas e inorgânicas encontradas na água.


<img align= center src="https://user-images.githubusercontent.com/83380670/142760252-c3330e3f-300c-45e0-9042-49dffaae5886.png" width="70%" height="70%" />

Cloraminas são um grupo de subprodutos de desinfecção, formado a partir da reação de cloro com o nitrogênio da amônia. Podendo ser derivados de procedimentos de tratamento da água e também de detritos celulares de bactérias e algas mortas  A mono-cloramina é o membro mais simples e mais comum do grupo, muitas vezes produzido intencionalmente a partir da reação de cloro puro e amônia pura.
São consideradas contaminantes por serem tóxicas e terem um odor desagradável 


<img align= center src="https://user-images.githubusercontent.com/83380670/142760272-16dbe655-8742-4d3c-9230-720a82e684ec.png" width="70%" height="70%" />

O sulfato (SO4) pode ser encontrado em quase todas as águas naturais, sendo um dos íons mais abundantes na água marinha e estuarina. A origem da maioria dos compostos de sulfato é a oxidação de minérios de sulfito, a presença de xisto ou resíduos industriais. O sulfato pode ser altamente prejudicial a saúde humana, causando diarréias, vômitos e demais problemas de saúde, caso esteja presente na água em excesso. 

<img align= center src="https://user-images.githubusercontent.com/83380670/142760283-d16e6537-1f5a-404d-8b28-d0115c108748.png" width="70%" height="70%" />

A Condutividade elétrica da água é a capacidade que a água possui de conduzir corrente elétrica. Este parâmetro está relacionado com a presença de íons dissolvidos na água, por serem partículas carregadas eletricamente, quanto maior for a quantidade de íons dissolvidos, maior será a condutividade elétrica na água.


<img align= center src="https://user-images.githubusercontent.com/83380670/143085387-903dfb45-e4d8-46e4-a80a-a8163335ccc7.png" width="70%" height="70%" />


Carbono orgânico total é a medida da quantidade total de carbono de compostos orgânicos. Esses compostos podem ser provenientes de fontes naturais, como a decomposição de material orgânico ou contaminantes como inseticidas, herbicidas, produtos agrícolas que entram no curso de água pelo escoamento da chuva. Outra fonte de carbono orgânico muito comum é através de águas residuais domésticas e industriais que são dispejados nos corpos dágua.


<img align= center src="https://user-images.githubusercontent.com/83380670/143085577-c2c1c013-a8c1-4be2-86c5-e6b4021b2759.png" width="70%" height="70%" />


Os trialometanos constituem um grupo de compostos orgânicos derivados do metano (CH4), substituídos por halogênios e possuem a fórmula geral CHX3, onde X pode ser cloro, bromo, possivelmente iodo, ou combinações a partir dos mesmos. Ocorrem, principalmente na água potável, como produtos resultantes da reação entre substâncias químicas que se utilizam em tratamentos oxidativos.


<img align= center src="https://user-images.githubusercontent.com/83380670/142760341-e0da65f2-dd92-486a-ad94-43bfbbdfc1c0.png" width="70%" height="70%" />

A turbidez é a medida da dificuldade de um feixe de luz atravessar uma certa quantidade de uma solução, os valores dessa medida são expressos, normalmente, em Unidades Nefelométricas de Turbidez - UNT. 
Na água suas principais causas da turbidez são: presença de matérias sólidas em suspensão (silte, argila, sílica, coloides), matéria orgânica e inorgânica finamente divididas, organismos microscópicos e algas.
A turbidez reduz a penetração da luz solar na coluna d´água, prejudicando a fotossíntese das algas e plantas aquáticas submersas.

<img align= center src="https://user-images.githubusercontent.com/83380670/142760364-53ff85aa-2f25-49bd-bb3c-6b9aa2c5c8e5.png" width="70%" height="70%" />

### Análise preditiva: 

Após feito o tratamento de dados necessário afim de normalização dos registros , os dados foram divididos em dados de treino e dados de testes e feito então a aplicação dos modelos ao conjunto de dados de treino para ajuste dos parâmetros e aos dados de teste para verificar a precisão das predições do modelo. 

<img align= center src="https://user-images.githubusercontent.com/83380670/143253861-9ecf1820-941d-4e5c-afac-3be27bcc4902.png" width="30%" height="70%" />


Após treino dos modelos, foi feito também a otimização dos hiperparâmetros dos três modelos que apresentaram melhor performance. 

<img align= center src="https://user-images.githubusercontent.com/83380670/143309627-7152dc24-5cb2-432d-8a7b-0e37890b0d54.png" width="80%" height="70%" />

#### Modelo Final

O modelo que apresentou melhor performance  foi a Regressão Logística, com uma precisão de 69% 
Vale ressaltar a presença de outliers nas amostras e que foram incluídos na modelagem. Um tratamento adequado a esses dados é fundamental no treinamento do modelo.



## Referência Bibliográfica

[1] SWAN. Qualidade da água. Disponível em <https://tratamentodeagua.com.br/artigo/qualidade-da-agua/> Acesso em 14 de nov de 2021

[2] ANA.GOV. Qualidade da água. Disponível em: <https://www.gov.br/ana/pt-br/assuntos/monitoramento-e-eventos-criticos/qualidade-da-agua>.Acesso em 14 de nov de 2021

[3] SAAEC.Qualidade da água. Disponível em: <https://www.saaec.com.br/agua/qualidade-da-agua/>.Acesso em 13 de nov de 2021

[4]. ASSTEFIL. Redução de Cloraminas. Disponível em: <https://asstefil.com.br/solucoes-tratamento-filtragem-agua/reducao-de-cloraminas/>.Acesso em 20 de nov de 2021

[5] SABESP. Qualidade da água tratada. Disponível em: <http://site.sabesp.com.br/site/interna/Default.aspx?secaoId=40>.Acesso em 21 de nov de 2021.


[6] APDA. Dureza total. Disponível em <https://www.apda.pt/site/upload/FT-QI-10-%20Dureza%20total.pdf> Acesso em 22 de nov de 2021.

[6] DIGITALWATER.Sólidos Dissolvidos e Purificação De Água.Disponível em: <https://www.digitalwater.com.br/solidos-dissolvidos-filtracao-da-agua/> Acesso em 21 de nov. de 2021

[7] LABOPRIME.Análise de sulfato em água, o que é e como funciona?.Disponível em: <https://www.laboprime.com.br/analise-de-sulfato-em-agua-e-efluentes/> Acesso em 20 de nov de 2021

[8] CABRERA.L.C, BAUMGARTEN.M.G.Z et al. Adaptação do Método Turbidimétrico para análie de sulfato em amostras de águas estauarinas e marinhas.Repositório FURG.7-10.2006

[9] DIGITALWATER.Condutividade Elétrica: Guia Definitivo.Disponível em: <https://www.digitalwater.com.br/condutividade-eletrica-guia-definitivo/> Acesso em 20 de nov. de 2021

[10] ELGA.Total Organic Carbon (TOC) and its Measurement.Disponível em: <https://www.elgalabwater.com/blog/total-organic-carbon-toc>.Acesso em 22 de nov de 2021

[11] MIDIO.A.F,TOMINAGA.M.Y.Exposição humana a trialometanos presentes em água tratada.Revista de Saúde Pública. 1999

[12] SANTOS, Carlos Lopes. Trihalometanos: resumo atual. Engenharia Sanitária, v. 26, p. 190-194, jun. 1987.

