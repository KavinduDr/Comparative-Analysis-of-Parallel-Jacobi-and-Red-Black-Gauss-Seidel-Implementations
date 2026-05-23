import matplotlib.pyplot as plt
import numpy as np

labels = [
    'Serial', 
    'OpenMP (2t)', 'OpenMP (4t)', 
    'Pthreads (2t)', 'Pthreads (4t)', 
    'MPI (2p)', 'MPI (4p)', 
    'Hybrid CPU', 
    'CUDA', 'Hybrid CUDA'
]

jacobi_times = [
    0.584962, 0.392477, 0.432614, 0.615691, 0.976714, 
    0.322156, 0.276896, 0.498103, 1.096786, 3.424722
]

rbgs_times = [
    0.560960, 0.465577, 0.343684, 0.818879, 1.048269, 
    0.958363, 0.565029, 0.941881, 1.156729, 5.710918
]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))
rects1 = ax.bar(x - width/2, jacobi_times, width, label='Jacobi', color='#3498db')
rects2 = ax.bar(x + width/2, rbgs_times, width, label='Red-Black GS', color='#e74c3c')

ax.set_ylabel('Execution Time (seconds)', fontsize=12)
ax.set_title('Execution Time Comparison on 200x200 Grid (5000 Iterations)', fontsize=14, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.legend(fontsize=12)

ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords='offset points',
                    ha='center', va='bottom', fontsize=9, rotation=90)

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig('performance_comparison_chart.png', dpi=300, bbox_inches='tight')
print("Successfully generated performance_comparison_chart.png")
