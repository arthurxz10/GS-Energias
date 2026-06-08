# Missão ORION — Monitoramento Energético

**Global Solution 2026.1 | Energias Renováveis e Sustentáveis | FIAP 1CCPY**

---

## Grupo 05

| Nome | RM |
|------|----|
| Arthur dos Santos Bezerra | 569721 |
| Carlos Henrique Fratezi | 571792 |
| Felipe Gouveia Braga | 568956 |

---

## Sobre o projeto

A ideia do projeto é simular o sistema de monitoramento energético da cápsula espacial **ORION**, usando dados reais de exoplanetas da NASA para avaliar o potencial energético de diferentes sistemas estelares.

A cápsula possui painéis solares de alta eficiência (GaAs, 29%) e precisa alimentar 5 módulos operacionais. O sistema analisa cada estrela-alvo, calcula quanta energia os painéis conseguiriam captar naquele sistema, e classifica automaticamente o nível de alerta da missão.

**Dataset:** [NASA Exoplanet Archive — PSCompPars](https://exoplanetarchive.ipac.caltech.edu)  
Colunas usadas: temperatura estelar (`st_teff`), luminosidade estelar (`st_lum`), raio estelar (`st_rad`), temperatura de equilíbrio do planeta (`pl_eqt`).

---

## Funcionalidades

- **Monitoramento energético** — calcula irradiância, potência captada e eficiência para cada sistema estelar usando a Lei do Inverso do Quadrado
- **Alertas automáticos** — classifica cada sistema em CRÍTICO, ATENÇÃO ou NOMINAL com ação recomendada
- **IA introdutória** — regressão linear (scikit-learn) treinada para prever potência com base em luminosidade + temperatura (R² = 0.877)
- **5 gráficos** gerados automaticamente na pasta `orion_outputs/`
- **Relatório final** em `.txt` com estatísticas completas

---

## Módulos da cápsula ORION

| Módulo | Função | Consumo |
|--------|--------|---------|
| ORION-COMM | Comunicação com a Terra | 1.800 W |
| ORION-LIFE | Suporte à vida | 3.200 W |
| ORION-NAV | Navegação e propulsão | 950 W |
| ORION-SCI | Instrumentos científicos | 2.100 W |
| ORION-THERM | Controle térmico | 1.400 W |
| **TOTAL** | | **9.450 W** |

---

## Como rodar

**1. Instalar as dependências:**
```bash
pip install pandas numpy matplotlib scikit-learn
```

**2. Executar:**
```bash
python3 orion_energias.py
```

**3. Os outputs são salvos automaticamente em `orion_outputs/`:**
```
orion_outputs/
├── 01_potencia_por_estrela.png
├── 02_diagrama_hr.png
├── 03_balanco_energetico.png
├── 04_regressao_ia.png
├── 05_alertas_pizza.png
└── relatorio_final.txt
```

Não precisa instalar nada além das bibliotecas — o dataset já está embutido no código.

---

## Gráficos gerados

### Potência captada por estrela-alvo
Barras coloridas por nível de alerta, com linha de referência do consumo total da cápsula.

![Potência por estrela](orion_outputs/01_potencia_por_estrela.png)

### Diagrama HR simplificado
Temperatura × luminosidade dos sistemas estelares analisados, colorido por tipo espectral.

![Diagrama HR](orion_outputs/02_diagrama_hr.png)

### Balanço energético
Mostra quais sistemas têm superávit (verde) e quais têm déficit (vermelho).

![Balanço](orion_outputs/03_balanco_energetico.png)

### Regressão Linear — Real vs Previsto
Compara a potência real calculada com a potência prevista pelo modelo de IA.

![Regressão](orion_outputs/04_regressao_ia.png)

### Distribuição dos alertas
Pizza com a proporção de sistemas em cada nível de alerta.

div align="center"
img src="https://github.com/user-attachments/assets/8ec23c49-8e74-4dff-9b87-43f708ec310c" width="0px" /
/div

---

## Resultados principais

- **50 exoplanetas** analisados de **37 sistemas estelares**
- O sistema **Kepler-452** (estrela tipo G, similar ao Sol) é o mais eficiente — gera **37.465 W**, com superávit de +28.015 W
- **TRAPPIST-1** e **Proxima Centauri** (anãs M) entram em estado crítico por baixa luminosidade
- O modelo de IA atingiu **R² = 0.877**, explicando 87,7% da variância na potência captada

---

## Tecnologias

- Python 3
- pandas — manipulação dos dados
- numpy — cálculos físicos
- matplotlib — geração dos gráficos
- scikit-learn — regressão linear (IA introdutória)

---

## Contexto acadêmico

Projeto desenvolvido para a disciplina de **Energias Renováveis e Sustentáveis** como parte do tema integrador **Missão ORION** da Global Solution 2026.1 — FIAP.

> *"A maioria dos sistemas está em estado crítico porque o dataset é composto principalmente de anãs M e K, que têm baixa luminosidade. Isso significa que a cápsula não captaria energia suficiente pra manter todos os módulos funcionando a 1 UA dessas estrelas."*
