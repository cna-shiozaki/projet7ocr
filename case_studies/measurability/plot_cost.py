import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm
from matplotlib.ticker import FuncFormatter

C_NB_REPAY_SUCCESS = int(sys.argv[ sys.argv.index("-success") + 1])
C_NB_REPAY_FAILURE = int(sys.argv[ sys.argv.index("-failure") + 1])

# Profit moyen pour un client qui rembourse bien son prêt
C_AVG_PROFIT = 15000

# Ratio moyen entre client qui rembourse son prêt et défaut de paiement
# (un défaut coûte C_PROFIT_LOSS_RATIO fois ce que la banque gagne avec un client qui rembourse son prêt)
C_PROFIT_LOSS_RATIO = 10


x_ = np.linspace(0., C_NB_REPAY_SUCCESS, 100)
y_ = np.linspace(0, C_NB_REPAY_FAILURE, 100)

x, y = np.meshgrid(x_, y_)

z = ( C_AVG_PROFIT * ( x - C_PROFIT_LOSS_RATIO * y )  ) / 10**6


fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(15,15))

surface = ax.plot_surface(x, y, z, linewidth=0, antialiased=False, cmap=cm.RdYlGn)
ax.set_xlabel('Nombre de clients qui remboursent leur prêt', fontsize=10, rotation=150)
ax.set_ylabel('Nombre de clients qui font défaut', fontsize=10, rotation=150)
ax.set_zlabel('Bénéfice (en million)', fontsize=20, rotation=150)

ax.get_xaxis().set_major_formatter( FuncFormatter(lambda x, p: format(int(x), ' ')))
ax.zaxis.set_major_formatter( FuncFormatter(lambda x, p: format(int(x), ' ')))


plt.show()