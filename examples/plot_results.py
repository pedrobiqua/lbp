import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Carrega a matriz de confusão do arquivo CSV
df = pd.read_csv("build/confusion_matrix.csv", index_col=0)

# Cria a figura
plt.figure(figsize=(6, 5))

# Plota a matriz de confusão com uma paleta de cores "Blues"
sns.heatmap(df, annot=True, fmt="d", cmap="Blues", cbar=True)

# Títulos e rótulos
plt.title("Matriz de Confusão - KNN + LBP")
plt.xlabel("Predito")
plt.ylabel("Verdadeiro")

# Exibe o gráfico
plt.savefig("confusion_matrix.png")
plt.show()
