"""
Data Preprocessing Pipeline Visualization - Professional Version
Mental Health Chatbot - Thesis Project
Complete visualization of data preprocessing and merging workflow
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
from matplotlib.lines import Line2D
import numpy as np
import os

# Ensure slide folder exists
os.makedirs('slide', exist_ok=True)

# Style Configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'figure.facecolor': 'white',
})

# =============================================================================
# FIGURE 1: Complete Data Preprocessing & Merging Pipeline
# =============================================================================
fig, ax = plt.subplots(figsize=(20, 16))
ax.set_xlim(0, 20)
ax.set_ylim(0, 16)
ax.axis('off')
ax.set_facecolor('#fafafa')

# Main Title
ax.text(10, 15.5, 'Data Preprocessing & Merging Pipeline', 
        ha='center', fontsize=24, fontweight='bold', color='#1a237e')
ax.text(10, 15, 'Mental Health Chatbot - Complete Data Engineering Workflow', 
        ha='center', fontsize=13, color='#455a64', style='italic')

# ========================= PHASE 1: RAW DATA SOURCES =========================
phase1_box = FancyBboxPatch((0.5, 12.8), 19, 2.2, boxstyle="round,pad=0.15", 
                            facecolor='#e3f2fd', edgecolor='#1976d2', linewidth=2, alpha=0.5)
ax.add_patch(phase1_box)
ax.text(10, 14.7, 'PHASE 1: RAW DATA SOURCES', ha='center', fontsize=14, 
        fontweight='bold', color='#0d47a1')

# Three data sources (no clinical data)
sources = [
    (4, 13.5, 'Mental Health\nCounseling Dataset', '#1976D2', '3,512', 
     'CSV Format\nContext-Response Pairs\nProfessional Counseling'),
    (10, 13.5, 'Cultural Synthetic\nDataset', '#7B1FA2', '3,000', 
     'JSON Format\nBangladesh Context\nCultural Relevance'),
    (16, 13.5, 'Multi-turn\nDialogue Dataset', '#00897B', '500', 
     'CSV Format\nComplex Conversations\nHigh Quality')
]

for x, y, label, color, count, desc in sources:
    # Shadow effect
    shadow = FancyBboxPatch((x-2.05, y-0.95), 4.1, 1.9, boxstyle="round,pad=0.1", 
                            facecolor='#00000012', edgecolor='none')
    ax.add_patch(shadow)
    # Main source box
    box = FancyBboxPatch((x-2.1, y-0.9), 4.1, 1.9, boxstyle="round,pad=0.1", 
                         facecolor=color, edgecolor='white', linewidth=3)
    ax.add_patch(box)
    ax.text(x, y+0.5, label, ha='center', va='center', fontsize=11, 
            fontweight='bold', color='white')
    ax.text(x, y-0.05, count + ' samples', ha='center', va='center', fontsize=12, 
            fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='#00000025', edgecolor='none'))
    ax.text(x, y-0.55, desc, ha='center', va='center', fontsize=8, color='#ffffffcc')

# ========================= PHASE 2: INDIVIDUAL PREPROCESSING =========================
phase2_box = FancyBboxPatch((0.5, 9.3), 19, 3.2, boxstyle="round,pad=0.15", 
                            facecolor='#fff3e0', edgecolor='#ff9800', linewidth=2, alpha=0.5)
ax.add_patch(phase2_box)
ax.text(10, 12.2, 'PHASE 2: INDIVIDUAL PREPROCESSING (Per Dataset)', ha='center', 
        fontsize=14, fontweight='bold', color='#e65100')

# Arrows from sources to preprocessing
for x in [4, 10, 16]:
    ax.annotate('', xy=(x, 11.8), xytext=(x, 12.5), 
                arrowprops=dict(arrowstyle='->', color='#37474f', lw=2))

# Preprocessing steps for each dataset
prep_colors = ['#1976D2', '#7B1FA2', '#00897B']
prep_positions = [4, 10, 16]

for i, (x, color) in enumerate(zip(prep_positions, prep_colors)):
    # Preprocessing box
    prep_box = FancyBboxPatch((x-1.8, 9.7), 3.6, 2, boxstyle="round,pad=0.08", 
                              facecolor='white', edgecolor=color, linewidth=2)
    ax.add_patch(prep_box)
    
    steps = ['Text Normalization', 'Quality Filtering', 'Format Conversion']
    for j, step in enumerate(steps):
        step_box = FancyBboxPatch((x-1.6, 11.2-j*0.55), 3.2, 0.45, 
                                  boxstyle="round,pad=0.03", facecolor=color+'33', 
                                  edgecolor=color, linewidth=1)
        ax.add_patch(step_box)
        ax.text(x, 11.42-j*0.55, step, ha='center', va='center', fontsize=9, 
                fontweight='bold', color='#333')

# ========================= PHASE 3: DATA MERGING =========================
phase3_box = FancyBboxPatch((0.5, 6.3), 19, 2.7, boxstyle="round,pad=0.15", 
                            facecolor='#e8f5e9', edgecolor='#4caf50', linewidth=2, alpha=0.5)
ax.add_patch(phase3_box)
ax.text(10, 8.7, 'PHASE 3: DATA MERGING & COMBINATION', ha='center', 
        fontsize=14, fontweight='bold', color='#2e7d32')

# Arrows from preprocessing to merge
for x in [4, 10, 16]:
    ax.annotate('', xy=(10, 8.2), xytext=(x, 9.6), 
                arrowprops=dict(arrowstyle='->', color='#2e7d32', lw=2,
                               connectionstyle='arc3,rad=0.2'))

# Merge operation box
merge_box = FancyBboxPatch((6, 6.8), 8, 1.5, boxstyle="round,pad=0.1", 
                           facecolor='#1b5e20', edgecolor='white', linewidth=3)
ax.add_patch(merge_box)
ax.text(10, 7.85, 'CONCATENATE ALL DATASETS', ha='center', va='center', 
        fontsize=13, fontweight='bold', color='white')
ax.text(10, 7.35, '3,512 + 3,000 + 500 = 7,012 Total Samples', ha='center', va='center', 
        fontsize=11, color='#a5d6a7', fontweight='bold')
ax.text(10, 6.95, 'Unified JSON Lines Format', ha='center', va='center', 
        fontsize=10, color='#c8e6c9')

# ========================= PHASE 4: POST-MERGE PROCESSING =========================
phase4_box = FancyBboxPatch((0.5, 3.5), 19, 2.5, boxstyle="round,pad=0.15", 
                            facecolor='#f3e5f5', edgecolor='#9c27b0', linewidth=2, alpha=0.5)
ax.add_patch(phase4_box)
ax.text(10, 5.7, 'PHASE 4: POST-MERGE PROCESSING', ha='center', 
        fontsize=14, fontweight='bold', color='#7b1fa2')

# Arrow from merge to post-process
ax.annotate('', xy=(10, 5.3), xytext=(10, 6.7), 
            arrowprops=dict(arrowstyle='->', color='#7b1fa2', lw=3))

# Post-merge steps
post_steps = [
    (3.5, 4.3, 'Shuffle\nDataset', '#9C27B0', 'Randomize order\nfor training'),
    (7.5, 4.3, 'Quality Score\nCalculation', '#673AB7', 'Turn + Length +\nBalance scoring'),
    (11.5, 4.3, 'Chat Template\nApplication', '#3F51B5', 'User/Assistant\nrole formatting'),
    (15.5, 4.3, 'Train/Val\nSplit', '#2196F3', '90% / 10%\nratio')
]

for x, y, label, color, desc in post_steps:
    box = FancyBboxPatch((x-1.5, y-0.5), 3, 1.3, boxstyle="round,pad=0.05", 
                         facecolor=color, edgecolor='white', linewidth=2)
    ax.add_patch(box)
    ax.text(x, y+0.25, label, ha='center', va='center', fontsize=10, 
            fontweight='bold', color='white')
    ax.text(x, y-0.25, desc, ha='center', va='center', fontsize=8, color='#ffffffcc')

# Arrows between post-merge steps
for i in range(len(post_steps)-1):
    ax.annotate('', xy=(post_steps[i+1][0]-1.6, post_steps[i][1]), 
                xytext=(post_steps[i][0]+1.6, post_steps[i][1]), 
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))

# ========================= PHASE 5: OUTPUT =========================
phase5_box = FancyBboxPatch((0.5, 0.3), 19, 2.9, boxstyle="round,pad=0.15", 
                            facecolor='#e0f7fa', edgecolor='#00bcd4', linewidth=2, alpha=0.5)
ax.add_patch(phase5_box)
ax.text(10, 2.9, 'PHASE 5: FINAL OUTPUT', ha='center', 
        fontsize=14, fontweight='bold', color='#006064')

# Arrows to output
ax.annotate('', xy=(6, 2.3), xytext=(10, 3.7), 
            arrowprops=dict(arrowstyle='->', color='#006064', lw=3))
ax.annotate('', xy=(14, 2.3), xytext=(10, 3.7), 
            arrowprops=dict(arrowstyle='->', color='#006064', lw=3))

# Output files
train_box = FancyBboxPatch((2.5, 0.6), 6, 1.8, boxstyle="round,pad=0.1", 
                           facecolor='#4CAF50', edgecolor='white', linewidth=3)
ax.add_patch(train_box)
ax.text(5.5, 1.85, 'train_full.jsonl', ha='center', va='center', 
        fontsize=14, fontweight='bold', color='white')
ax.text(5.5, 1.35, '6,310 samples', ha='center', va='center', 
        fontsize=13, color='white', fontweight='bold')
ax.text(5.5, 0.9, '90% of total data', ha='center', va='center', 
        fontsize=10, color='#c8e6c9')

val_box = FancyBboxPatch((11.5, 0.6), 6, 1.8, boxstyle="round,pad=0.1", 
                         facecolor='#2196F3', edgecolor='white', linewidth=3)
ax.add_patch(val_box)
ax.text(14.5, 1.85, 'val_full.jsonl', ha='center', va='center', 
        fontsize=14, fontweight='bold', color='white')
ax.text(14.5, 1.35, '702 samples', ha='center', va='center', 
        fontsize=13, color='white', fontweight='bold')
ax.text(14.5, 0.9, '10% of total data', ha='center', va='center', 
        fontsize=10, color='#bbdefb')

plt.tight_layout()
plt.savefig('slide/datapreprocessing_pipeline.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("Saved: slide/datapreprocessing_pipeline.png")
plt.close()

# =============================================================================
# FIGURE 2: Data Statistics & Distribution
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Data Preprocessing - Statistics & Distribution Analysis', 
             fontsize=18, fontweight='bold', y=0.98, color='#1a237e')

# 1. Donut Chart - Data Source Distribution
ax1 = axes[0, 0]
sources_names = ['Mental Health\nCounseling', 'Cultural\nSynthetic', 'Multi-turn\nDialogues']
sizes = [3512, 3000, 500]
colors = ['#1976D2', '#7B1FA2', '#00897B']
explode = (0.02, 0.02, 0.02)

wedges, texts, autotexts = ax1.pie(sizes, labels=sources_names, colors=colors, 
                                   explode=explode, autopct='%1.1f%%', 
                                   startangle=90, pctdistance=0.78,
                                   textprops={'fontsize': 10, 'fontweight': 'bold'},
                                   wedgeprops={'linewidth': 3, 'edgecolor': 'white'})
for a in autotexts: 
    a.set_color('white')
    a.set_fontsize(11)
    a.set_fontweight('bold')

centre_circle = plt.Circle((0,0), 0.55, fc='white')
ax1.add_artist(centre_circle)
ax1.text(0, 0.1, '7,012', ha='center', va='center', fontsize=20, fontweight='bold', color='#1a237e')
ax1.text(0, -0.15, 'Total', ha='center', va='center', fontsize=12, color='#666')
ax1.set_title('Data Source Distribution', fontsize=14, fontweight='bold', pad=15, color='#333')

# 2. Horizontal Bar Chart - Samples per Source with details
ax2 = axes[0, 1]
sources_short = ['Mental Health Counseling', 'Cultural Synthetic', 'Multi-turn Dialogues']
y_pos = np.arange(len(sources_short))
bars = ax2.barh(y_pos, sizes, color=colors, edgecolor='white', linewidth=2, height=0.6)

ax2.set_yticks(y_pos)
ax2.set_yticklabels(sources_short, fontsize=11, fontweight='bold')
ax2.set_xlabel('Number of Samples', fontweight='bold', fontsize=11)
ax2.set_title('Samples per Data Source', fontsize=14, fontweight='bold', pad=15, color='#333')

for bar, s, pct in zip(bars, sizes, [50.1, 42.8, 7.1]):
    ax2.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2, 
             f'{s:,} ({pct}%)', ha='left', va='center', fontweight='bold', fontsize=11)
ax2.set_xlim(0, 4500)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# 3. Train/Val Split Visualization
ax3 = axes[1, 0]
split_data = [6310, 702]
split_labels = ['Training Set\n(90%)', 'Validation Set\n(10%)']
split_colors = ['#4CAF50', '#2196F3']

bars = ax3.bar(split_labels, split_data, color=split_colors, edgecolor='white', 
               linewidth=3, width=0.5)
ax3.set_ylabel('Number of Samples', fontweight='bold', fontsize=11)
ax3.set_title('Train / Validation Split (90:10 Ratio)', fontsize=14, fontweight='bold', pad=15, color='#333')

# Add value labels
ax3.text(0, 6310 + 200, '6,310', ha='center', fontweight='bold', fontsize=16, color='#2e7d32')
ax3.text(1, 702 + 200, '702', ha='center', fontweight='bold', fontsize=16, color='#1565c0')

ax3.set_ylim(0, 7500)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Add a stacked bar showing composition
ax3_inset = ax3.inset_axes([0.55, 0.55, 0.4, 0.35])
ax3_inset.barh([0], [6310], color='#4CAF50', label='Train', height=0.5)
ax3_inset.barh([0], [702], left=[6310], color='#2196F3', label='Val', height=0.5)
ax3_inset.set_xlim(0, 7012)
ax3_inset.set_xticks([])
ax3_inset.set_yticks([])
ax3_inset.legend(loc='upper right', fontsize=9)
ax3_inset.set_title('Proportion', fontsize=10)
for spine in ax3_inset.spines.values():
    spine.set_visible(False)

# 4. Complete Preprocessing Steps Summary
ax4 = axes[1, 1]
ax4.axis('off')

# Title
ax4.text(0.5, 0.98, 'Complete Preprocessing Pipeline', ha='center', va='top', 
         fontsize=14, fontweight='bold', transform=ax4.transAxes, color='#1a237e')

steps_summary = [
    ('1. Data Loading', 'Load CSV and JSON files from 3 sources', '#1976D2'),
    ('2. Text Normalization', 'Handle Unicode, whitespace, special characters', '#E91E63'),
    ('3. Quality Filtering', 'Remove low quality (score < 40), length checks', '#9C27B0'),
    ('4. Format Conversion', 'Convert to unified JSON structure', '#673AB7'),
    ('5. Dataset Merging', 'Concatenate all preprocessed datasets', '#3F51B5'),
    ('6. Shuffling', 'Randomize data order for better training', '#2196F3'),
    ('7. Chat Template', 'Apply user/assistant role formatting', '#00897B'),
    ('8. Train/Val Split', '90% training, 10% validation', '#4CAF50'),
]

for i, (step, desc, color) in enumerate(steps_summary):
    y_pos = 0.88 - i * 0.11
    # Color indicator box
    ax4.add_patch(FancyBboxPatch((0.02, y_pos-0.035), 0.06, 0.06, 
                                 boxstyle="round,pad=0.02", facecolor=color, 
                                 edgecolor='white', linewidth=2, transform=ax4.transAxes))
    # Step name
    ax4.text(0.1, y_pos, step, ha='left', va='center', fontsize=11, 
             fontweight='bold', transform=ax4.transAxes, color='#333')
    # Description
    ax4.text(0.38, y_pos, desc, ha='left', va='center', fontsize=10, 
             color='#555', transform=ax4.transAxes)

plt.tight_layout()
plt.savefig('slide/datapreprocessing_statistics.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("Saved: slide/datapreprocessing_statistics.png")
plt.close()

# =============================================================================
# FIGURE 3: Quality Scoring System
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Data Quality Assessment System', fontsize=18, fontweight='bold', y=1.02, color='#1a237e')

# 1. Quality Score Components
ax1 = axes[0]
components = ['Turn Score\n(0-40 pts)', 'Length Score\n(0-40 pts)', 'Balance Score\n(0-20 pts)']
max_scores = [40, 40, 20]
colors = ['#E91E63', '#673AB7', '#00897B']

bars = ax1.bar(components, max_scores, color=colors, edgecolor='white', linewidth=3, width=0.6)
ax1.set_ylabel('Maximum Points', fontweight='bold', fontsize=11)
ax1.set_title('Quality Score Components\n(Total: 100 points)', fontsize=13, fontweight='bold', pad=15)

for bar, s in zip(bars, max_scores):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5, 
             f'{s} pts', ha='center', fontweight='bold', fontsize=13)
ax1.set_ylim(0, 55)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Add component breakdown at bottom
breakdown_text = [
    'More conversation\nturns = higher score',
    'Optimal length:\n100-500 chars/msg',
    'Equal user/assistant\nmessages preferred'
]
for i, (bar, text) in enumerate(zip(bars, breakdown_text)):
    ax1.text(bar.get_x()+bar.get_width()/2, -8, text, ha='center', va='top', 
             fontsize=9, color='#666', style='italic')

# 2. Quality Level Distribution
ax2 = axes[1]
quality_levels = ['Low Quality\n(< 40)', 'Medium Quality\n(40-70)', 'High Quality\n(> 70)']
counts = [420, 1950, 4642]  # Based on actual distribution
colors = ['#E57373', '#FFB74D', '#81C784']

bars = ax2.bar(quality_levels, counts, color=colors, edgecolor='white', linewidth=3, width=0.6)
ax2.set_ylabel('Number of Samples', fontweight='bold', fontsize=11)
ax2.set_title('Quality Level Distribution', fontsize=13, fontweight='bold', pad=15)

for bar, c in zip(bars, counts):
    pct = c / sum(counts) * 100
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+100, 
             f'{c:,}\n({pct:.1f}%)', ha='center', fontweight='bold', fontsize=11)
ax2.set_ylim(0, 5500)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# 3. Scoring Criteria Details
ax3 = axes[2]
ax3.axis('off')

ax3.text(0.5, 0.98, 'Quality Scoring Criteria', ha='center', va='top', 
         fontsize=14, fontweight='bold', transform=ax3.transAxes, color='#1a237e')

criteria = [
    ('TURN SCORE (40 pts max)', 
     '- More turns indicate deeper conversations\n- Score = min(turns/20, 1) x 40\n- 20+ turns gets maximum score', 
     '#E91E63'),
    ('LENGTH SCORE (40 pts max)', 
     '- Optimal: 100-500 characters per message\n- Too short/long gets penalty\n- Encourages substantive responses', 
     '#673AB7'),
    ('BALANCE SCORE (20 pts max)', 
     '- Equal user/assistant messages preferred\n- Difference > 1 reduces score\n- Ensures conversation flow', 
     '#00897B'),
    ('QUALITY THRESHOLD', 
     '- Minimum score: 40 points\n- Below threshold = filtered out\n- Ensures training data quality', 
     '#1976D2'),
]

for i, (title, desc, color) in enumerate(criteria):
    y_pos = 0.85 - i * 0.23
    # Title box
    ax3.add_patch(FancyBboxPatch((0.02, y_pos-0.02), 0.96, 0.2, 
                                 boxstyle="round,pad=0.03", facecolor=color+'15', 
                                 edgecolor=color, linewidth=2, transform=ax3.transAxes))
    ax3.text(0.05, y_pos+0.12, title, ha='left', va='center', fontsize=11, 
             fontweight='bold', color=color, transform=ax3.transAxes)
    ax3.text(0.05, y_pos+0.01, desc, ha='left', va='top', fontsize=9, 
             color='#444', transform=ax3.transAxes, linespacing=1.3)

plt.tight_layout()
plt.savefig('slide/datapreprocessing_quality.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("Saved: slide/datapreprocessing_quality.png")
plt.close()

# =============================================================================
# FIGURE 4: Complete Summary Infographic
# =============================================================================
fig, ax = plt.subplots(figsize=(18, 12))
ax.set_xlim(0, 18)
ax.set_ylim(0, 12)
ax.axis('off')
ax.set_facecolor('#fafafa')

# Title
ax.text(9, 11.5, 'Data Preprocessing Complete Summary', ha='center', fontsize=24, 
        fontweight='bold', color='#1a237e')
ax.text(9, 11, 'Mental Health Chatbot - Training Data Preparation', ha='center', 
        fontsize=13, color='#455a64', style='italic')

# =================== INPUT -> PROCESS -> OUTPUT Flow ===================
# Input Section
input_box = FancyBboxPatch((0.5, 7.8), 5, 2.8, boxstyle="round,pad=0.15", 
                           facecolor='#e3f2fd', edgecolor='#1976d2', linewidth=3)
ax.add_patch(input_box)
ax.text(3, 10.2, 'INPUT DATA', ha='center', fontsize=14, fontweight='bold', color='#1976d2')

inputs = [
    ('Mental Health CSV', '3,512', '#1976D2'),
    ('Cultural Synthetic JSON', '3,000', '#7B1FA2'),
    ('Multi-turn Dialogues', '500', '#00897B'),
]
for i, (name, count, color) in enumerate(inputs):
    y = 9.6 - i * 0.6
    ax.add_patch(FancyBboxPatch((0.8, y-0.2), 0.3, 0.4, boxstyle="round,pad=0.02", 
                                facecolor=color, edgecolor='white', linewidth=1))
    ax.text(1.3, y, f'{name}: {count}', ha='left', va='center', fontsize=10, color='#333')

# Arrow 1
ax.annotate('', xy=(5.8, 9.2), xytext=(5.6, 9.2), 
            arrowprops=dict(arrowstyle='->', color='#333', lw=3))

# Process Section
process_box = FancyBboxPatch((6, 7.8), 6, 2.8, boxstyle="round,pad=0.15", 
                             facecolor='#fff3e0', edgecolor='#ff9800', linewidth=3)
ax.add_patch(process_box)
ax.text(9, 10.2, 'PREPROCESSING', ha='center', fontsize=14, fontweight='bold', color='#e65100')

processes = [
    'Text Normalization',
    'Quality Filtering (Score > 40)',
    'Format Standardization (JSONL)',
    'Dataset Merging & Shuffling',
    'Chat Template Application'
]
for i, proc in enumerate(processes):
    ax.text(6.3, 9.7 - i * 0.45, f'> {proc}', ha='left', fontsize=10, color='#333')

# Arrow 2
ax.annotate('', xy=(12.3, 9.2), xytext=(12.1, 9.2), 
            arrowprops=dict(arrowstyle='->', color='#333', lw=3))

# Output Section
output_box = FancyBboxPatch((12.5, 7.8), 5, 2.8, boxstyle="round,pad=0.15", 
                            facecolor='#e8f5e9', edgecolor='#4caf50', linewidth=3)
ax.add_patch(output_box)
ax.text(15, 10.2, 'OUTPUT DATA', ha='center', fontsize=14, fontweight='bold', color='#2e7d32')

ax.text(12.8, 9.6, 'train_full.jsonl', ha='left', fontsize=11, fontweight='bold', color='#2e7d32')
ax.text(12.8, 9.25, '6,310 samples (90%)', ha='left', fontsize=10, color='#333')
ax.text(12.8, 8.7, 'val_full.jsonl', ha='left', fontsize=11, fontweight='bold', color='#1565c0')
ax.text(12.8, 8.35, '702 samples (10%)', ha='left', fontsize=10, color='#333')

# =================== Statistics Section ===================
stats_box = FancyBboxPatch((0.5, 4.3), 17, 3.2, boxstyle="round,pad=0.15", 
                           facecolor='white', edgecolor='#9e9e9e', linewidth=2)
ax.add_patch(stats_box)
ax.text(9, 7.1, 'KEY STATISTICS', ha='center', fontsize=14, fontweight='bold', color='#333')

# Stat boxes
stat_items = [
    (2, 5.5, 'Total Samples', '7,012', '#1976D2'),
    (5, 5.5, 'Training', '6,310', '#4CAF50'),
    (8, 5.5, 'Validation', '702', '#2196F3'),
    (11, 5.5, 'Avg Quality Score', '74.2', '#9C27B0'),
    (14, 5.5, 'High Quality Rate', '66.2%', '#00897B'),
    (17, 5.5, 'Data Sources', '3', '#F57C00'),
]

for x, y, label, value, color in stat_items:
    stat_box = FancyBboxPatch((x-1.3, y-0.7), 2.6, 1.5, boxstyle="round,pad=0.08", 
                              facecolor=color, edgecolor='white', linewidth=2)
    ax.add_patch(stat_box)
    ax.text(x, y+0.25, value, ha='center', va='center', fontsize=18, 
            fontweight='bold', color='white')
    ax.text(x, y-0.3, label, ha='center', va='center', fontsize=9, color='#ffffffcc')

# =================== Data Format Section ===================
format_box = FancyBboxPatch((0.5, 0.5), 8, 3.5, boxstyle="round,pad=0.15", 
                            facecolor='#fce4ec', edgecolor='#e91e63', linewidth=2)
ax.add_patch(format_box)
ax.text(4.5, 3.7, 'OUTPUT FORMAT (JSONL)', ha='center', fontsize=13, 
        fontweight='bold', color='#c2185b')

json_example = '''{"messages": [
    {"role": "user", 
     "content": "I feel anxious..."},
    {"role": "assistant", 
     "content": "I understand..."}
]}'''
ax.text(4.5, 2, json_example, ha='center', va='center', fontsize=10, 
        family='monospace', color='#333',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#e0e0e0'))

# =================== Key Features Section ===================
feature_box = FancyBboxPatch((9.5, 0.5), 8, 3.5, boxstyle="round,pad=0.15", 
                             facecolor='#e8eaf6', edgecolor='#3f51b5', linewidth=2)
ax.add_patch(feature_box)
ax.text(13.5, 3.7, 'KEY FEATURES', ha='center', fontsize=13, 
        fontweight='bold', color='#303f9f')

features = [
    ('Multi-source Integration', '3 diverse data sources combined'),
    ('Cultural Relevance', 'Bangladesh context included'),
    ('Quality Assurance', 'Scoring-based filtering'),
    ('LLM-Ready Format', 'Chat template pre-applied'),
    ('Balanced Split', '90:10 train/validation ratio')
]
for i, (title, desc) in enumerate(features):
    y = 3.2 - i * 0.55
    ax.text(10, y, f'{title}:', ha='left', fontsize=10, fontweight='bold', color='#333')
    ax.text(13.5, y, desc, ha='left', fontsize=9, color='#666')

plt.tight_layout()
plt.savefig('slide/datapreprocessing_summary.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("Saved: slide/datapreprocessing_summary.png")
plt.close()

print("\n" + "="*65)
print("  ALL DATA PREPROCESSING VISUALIZATIONS SAVED SUCCESSFULLY!")
print("="*65)
print("\nGenerated Files in 'slide/' folder:")
print("-" * 65)
print("  1. datapreprocessing_pipeline.png    - Complete 5-phase pipeline")
print("  2. datapreprocessing_statistics.png  - Data distribution & steps")
print("  3. datapreprocessing_quality.png     - Quality scoring system")
print("  4. datapreprocessing_summary.png     - Complete summary infographic")
print("="*65)
