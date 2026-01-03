# 🏗️ System Architecture & Modules

当前 AlphaTwin 系统的模块化结构展示。

## 🧩 Module Interaction

```mermaid
graph TD
    subgraph Data_Factory [🏭 Data Factory]
        A[yfinance API] -->|Raw Data| B(Data Loader)
        B -->|Cleaning| C(Data Processor)
        C -->|Parquet| D[(Data Store)]
    end

    subgraph Strategy_Engine [🧠 Strategy Engine]
        D --> E[Backtest Engine]
        F[Strategy Logic] --> E
        E -->|Results| G[Performance Metrics]
    end

    subgraph Visualization [📊 Dashboard]
        G --> H[Plotly Charts]
        G --> I[Heatmap Generator]
    end

    style Data_Factory fill:#2d3436,stroke:#00b894,stroke-width:2px
    style Strategy_Engine fill:#2d3436,stroke:#0984e3,stroke-width:2px
    style Visualization fill:#2d3436,stroke:#e17055,stroke-width:2px