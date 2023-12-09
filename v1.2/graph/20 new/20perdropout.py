import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_json('history.json')

df1 = df

plt.plot(df1.index, df1['acc'], label='Training Accuracy')
plt.plot(df1.index, df1['val_acc'], label='Validation Accuracy')

plt.ylim(0.96, 1)

csfont = {'fontname': 'Times New Roman', 'fontsize': 14, 'fontweight': 'bold'}
plt.title('Model Accuracy and Loss with 20% Dropout', **csfont)
plt.ylabel('Accuracy / Loss', labelpad=15, **csfont)
plt.xlabel('Epoch', labelpad=15, **csfont)

plt.legend(prop={'family': 'Times New Roman', 'size': 12, 'weight': 'bold'})

plt.tick_params(axis='both', which='major', labelsize=12, width=2, length=6)

plt.xticks(fontname='Times New Roman')
plt.yticks(fontname='Times New Roman')

# plt.savefig('my_plot.png', dpi=300)

plt.show()
