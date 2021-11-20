
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



<!--
<img align= center src="https://user-images.githubusercontent.com/83380670/142292368-3fb7ffdf-7921-464d-ad24-363fb1c7b8e3.png" width="50%" height="70%" />
-->



Os dados após o tratamento de transformações de variáveis necessários foram divididos em dois conjuntos. Dados de treino com 471 registros e dados de teste com 118 registros e então feita a aplicação dos modelos ao conjunto de treino para ajuste dos parâmetros e posterior aplicação dos modelos aos dados de teste para verificar a concordância dos dados preditos com os dados reais e assim termos a precisão do modelo aplicado, descritos na tabela:

<img align= center src="https://user-images.githubusercontent.com/83380670/142728154-c169525b-d3fa-4015-92ae-2fb86d134d90.png" width="40%" height="70%" />


E no grafico:

<img align= center src="https://user-images.githubusercontent.com/83380670/142292534-ef2b7f50-aba9-432a-927e-a726b643484c.png" width="50%" height="70%" />

Para vizualizar o desempenho dos algorítimos, foi utilizada a construção da matriz de confusão ou matriz de erro , onde cada linha representa as instância da classe prevista e cada coluna as instâncias da classe atual.

<img align= center src="https://user-images.githubusercontent.com/83380670/142730444-691ed0d2-e35a-4e3e-94b3-0927c14d562c.png" width="40%" height="70%" />


Onde: P = Positivo; N = Negativo; VP = Verdadeiro Positivo; FP = Falso Positivo; VN = Verdadeiro Negativo; FN = Falso Negativo.


Todos os modelos predisseram a mesma quantidade de verdadeiros positivos, ou seja , quando a classe foi prevista corretamemte. Nenhum dos modelos apresentaram falsos positivos. Se diferenciaram na quantidade prevista para Falsos negativos e Verdadeiros Negativos.

<img align= center src="https://user-images.githubusercontent.com/83380670/142732444-e449d0b0-c0ac-42d4-87f9-531f8278b5bb.png" width="70%" height="70%" />





## Conclusão


A maioria das doenças hepáticas são inicialmente silenciosas. Por isso, os exames de rotina e a atenção a certos sinais do seu corpo são indispensáveis para manter a saúde deste órgão.
O estudo sistemático dos dados dos pacientes podem fornecer modelos para diagnósticos mais rápidos e precisos.
Quanto aos modelos de classificação empregados nesse projeto, a Regressão Logística e o CatBoosting apresentaram os maiores valores de precisão para predição dos diagnósticos de doenças hepáticas estudados. 

## Referência Bibliográfica
[1] CLINICACEU.Doenças hepáticas: tipos, sintomas e tratamentos.Disponível em, <https://www.clinicaceu.com.br/blog/doencas-hepaticas-tipos-sintomas-e-tratamentos/>.Acesso em: 18 de nov. de 2021

[2] HEPATOGASTRO.Doença Hepática.Disponível em <https://hepatogastro.com.br/doenca-hepatica/>Acesso em: 18 de Novembro de 2021

[3] CRAVEIRO, R.L. PLATAFORMA ON-LINE PARA O ENSINO DE BIOESTATÍSTICA: Ambiente virtual
de aprendizagem e software estatístico como ferramentas de apoio ao ensino.2016   

[4] DRAUSIO.Cirrose,Doenças e Sintomas.Disponível em <https://drauziovarella.uol.com.br/doencas-e-sintomas/cirrose/> Acesso em: 19 de Novembro de 2021

[5] BIBLIOTECA VIRTUAL EM SAÚDE. MS. Esteatose Hepática.Disponível em <https://bvsms.saude.gov.br/esteatose-hepatica/> Acesso em 19 de Novembro de 2021.

[6] CADERNO DE MEDICINA-UNIFESO.Perfil das doenças hepáticas crônicas no Ambulatório do UNIFESO. Disponível em <http://www.revista.unifeso.edu.br/index.php/cadernosdemedicinaunifeso/article/view/752>.Acesso em: 19 de Novembro de 2021
