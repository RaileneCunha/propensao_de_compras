# Propensão de Compras - Projeto de Machine Learning
Este projeto tem como objetivo prever a propensão de um cliente adquirir um serviço adicional de seguro automotivo,
utilizando algoritmos de aprendizado de máquina e disponibilizando uma API em Flask hospedada no Heroku.

## Problema de Negócio
Empresas de seguros buscam constantemente aumentar suas vendas de serviços adicionais. Com base no histórico de clientes,
este projeto propõe uma solução de classificação binária para identificar quais usuários têm maior probabilidade de aceitar a oferta de um seguro veicular.

## Objetivos
° Construir um modelo preditivo para estimar a propensão de compra.  

° Criar uma API REST que recebe dados de clientes e retorna a probabilidade de conversão.  

° Oferecer uma aplicação escalável que pode ser integrada a sistemas reais.  

## Técnicas e Ferramentas Utilizadas
° Python 3.13  

° Bibliotecas: Pandas, Scikit-learn, boruta, Flask, Request, Numpy, Pickle, Matplotlib, Healthinsurance, Seaborn  

° Machine Learning: Engenharia de atributos, tuning de hiperparâmetros  

° Deploy: Heroku (API), GitHub (código e notebooks)  

° Teste de API: Postman

## Como Usar
### 1. Clonar o repositório
```bash
git clone https://github.com/Railene_Cunha/propensao_de_compras.git
```
### 2. Instalar dependências
Crie e ative um ambiente virtual, depois:
```bash
pip install -r requirements.txt
```

### 3. Rodar a API localmente
```bash
python health_insurance_cross-sell/app.py
```
A API estará disponível em: `http://127.0.0.1:5000/heathinsurance/predict`

### 4. Fazer requisições (Exemplo com Postman ou script Python)

Envie uma requisição POST com um JSON contendo os dados dos clientes, por exemplo:

```json
[
  {
    "gender": "Male",
    "age": 35,
    "driving_license": 1,
    "region_code": 28,
    "previously_insured": 0,
    "vehicle_age": "1-2 Year",
    "vehicle_damage": "Yes",
    "annual_premium": 30000.0,
    "policy_sales_channel": 26,
    "vintage": 220
  }
]
```


## Diferenciais

- Pré-processamento modularizado em uma classe reutilizável.
- Modelo testado e validado com dados reais.
- API pronta para integração com sistemas externos.
- Código limpo, organizado e com foco em boas práticas de MLOps.

---

## Conclusão

Este projeto representa meus estudos e dedicação prática no uso de técnicas de Machine Learning aplicadas a um problema real.
Além do desenvolvimento do modelo, também explorei a importância de criar soluções completas, organizadas e integráveis através de uma API.
Ao hospedar a aplicação na nuvem e estruturar o código de forma clara, este trabalho reflete meu compromisso em aprender e aplicar boas práticas
desde o início da minha jornada na área de Ciência de Dados.
