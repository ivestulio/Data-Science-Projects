
<img align= center src="https://user-images.githubusercontent.com/83380670/142331746-32f9525d-30bd-427e-b884-bebb0ad0ab82.jpg" width="110%" height="100%" />



# Análise Preditiva - Doenças Hepáticas

O fígado é um órgão que exerce mais de 200 funções, além de ser responsável por manter o corpo livre de toxinas e células que se tornaram inúteis, auxiliar na digestão de alimentos, síntese de colesterol e metabolizar alguns elementos como bebidas alcoólicas e gorduras.
Doença hepática é qualquer condição que danifica o figado impedindo seu funxionamento e pode ter diferentes causas , seja genética ou decorrente de fatores que danificam o órgão, como vírus ou uso de alcool. 


## Objetivo

O objetivo desse projeto é gerar um modelo de predição de possível doença hepática usando como parâmetros de entrada o resultado laboratórial de sangue. Para isso será utilizado 7 modelos de classificação.

## Dataset
<h4>1. Resultados Lab. </h4>
O conjunto de dados contêm os resultados laboratóriais de doadores de sangue. 

**Fonte:**  https://archive.ics.uci.edu/ml/datasets/HCV+data
**File:**     <br>
**Registros:** <br>
**Parâmetros:** 
1) Identificação do Paciente
2) Diagnóstico
3) Idade
4) Sexo
5) ALB
6) ALP
7) ALT
8) AST
9) BIL
10) HE
11) CHOL
12) CREA
13) GGT
14) PROT



## Análise dos Dados

Os dados foram baixados do Kaggle e então analisados usando o Jupyter Nootebook.
Após a etapa de coleta dos dados, os mesmos foram tratados e explorados. Na etapa de tratamento foi possível fazer a limpeza e transformação dos dados, assim como criação de novas features possibilidando realizar vizualizações e entendimento da relação entre as variáveis


O projeto objetiva um modelo de predição do diagnóstico de saúde hepática a partir de um conjunto de observações de resultados laboratoriáis de sangue. Como o diagnóstico é uma variável categória podendo assumir (*após o tratamento de dados necessário*) 0 para saudável e 1 para doente. Foi utilizado nesse projetos modelos de classificação.


### Modelos
1. Logistic Regression
2. Decision Tree Classifier
3. Randon Forest Classifier
4. KNeighbors CLassifier
5. CatBoost Classifier
6. Gradient Boosting Classifier 


**O notebook em python com as análises feitas pode ser conferido em:** 


## Resultados

### Análise exploratória:

Inicialmente foi feita um estudo estatístico para conhecer a distribuição dos dados na amostra. 
Através do gráfico Boxplot é possível ver a presença de outiliers, em alguns casos muito acima do limite superior. Outliers são valores que destoam muito dos demais considerando para isso estar fora de um intervalo de distância interquartílica.   Um tratamento adequado aos outliers é fundamental para que eles não representem desvios no modelo de regressão e classificação. Nesse estudo em questão, por ser tratar de dados biológicos, não serão retirados do conjunto de dados e vamos observar como a precisão dos modelos que serão aplicados se comportam.

<img align= center src="https://user-images.githubusercontent.com/83380670/142464865-ce02cec4-36e5-4c64-9a5c-39769ea4b769.png" width="70%" height="70%" />

Na distribuição da contagem do número de pacientes por idade, se observa uma concentração maior de pacientes com idade próxima aos 45 anos, sendo o maio montante entre 30 e 65 anos de idade.

<img align= center src="https://user-images.githubusercontent.com/83380670/142291893-fbaf7086-3cab-4ca5-b1be-d6c76e852a84.png" width="70%" height="70%" />

Na distribuição do número de pacientes por sexo, sendo esses divididos entre saudáveis ou doentes( com alguma das doenças hepáticas listadas), nota-se a maior incidência de mulheres com doenças hepáticas em uma contagem total. Mesmo os homens sendo maioria nos casos de cirrose acoólica como pode ser conferido na literatura [4-6], as mulheres tem prevalência em esteatose hepáticas, também conhecida por doença hepática gordurosa.


<img align= center src="https://user-images.githubusercontent.com/83380670/142291938-5efffecc-71b2-4e72-bf2f-327502a538e3.png" width="70%" height="70%" />

Na distribuição dos resultados do exame de sangue por idade, identificado o diagnóstico, é possível observar a não presença de pessoas com diagnóstico de hepatite (0) para idades inferiores a 30 anos. Assim como a prevalência de pessoas com alguma doença hepática (1) acima dos 65.


<img align= center src="https://user-images.githubusercontent.com/83380670/142631728-967c633a-1f72-46ef-987e-e1e126b551f3.png" width="70%" height="70%" />

<img align= center src="https://user-images.githubusercontent.com/83380670/142292368-3fb7ffdf-7921-464d-ad24-363fb1c7b8e3.png" width="50%" height="70%" />


<img align= center src="https://user-images.githubusercontent.com/83380670/142292534-ef2b7f50-aba9-432a-927e-a726b643484c.png" width="50%" height="70%" />




## Conclusão
A maioria das doenças hepáticas são inicialmente silenciosas. Por isso, os exames de rotina e a atenção a certos sinais do seu corpo são indispensáveis para manter a saúde deste órgão.
## Referência Bibliográfica
[1] https://www.clinicaceu.com.br/blog/doencas-hepaticas-tipos-sintomas-e-tratamentos/

[2] https://hepatogastro.com.br/doenca-hepatica/

[3] produto-educacional-rogerio-lopes-craveiro.pdf

[4] https://drauziovarella.uol.com.br/doencas-e-sintomas/cirrose/
[5] https://bvsms.saude.gov.br/esteatose-hepatica/

[6] http://www.revista.unifeso.edu.br/index.php/cadernosdemedicinaunifeso/article/view/752
