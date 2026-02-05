"""
Mental Health Counseling Conversations - Category Analysis
Analyzes the dataset to identify topics/categories and creates visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import numpy as np

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('data/Mental_Health_Counseling_Conversations.csv')

print(f"\nDataset Overview:")
print(f"Total conversations: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# Keywords for different mental health categories
category_keywords = {
    'Depression': ['depress', 'sad', 'worthless', 'hopeless', 'crying', 'unmotivated', 'empty'],
    'Anxiety': ['anxiety', 'anxious', 'worry', 'worried', 'panic', 'fear', 'nervous', 'stress'],
    'Relationship Issues': ['relationship', 'marriage', 'divorce', 'partner', 'spouse', 'dating', 'breakup', 'family'],
    'Self-esteem': ['self-esteem', 'confidence', 'insecure', 'worthless', 'self-worth', 'self-image'],
    'Trauma/PTSD': ['trauma', 'ptsd', 'abuse', 'abused', 'assault', 'violence', 'traumatic'],
    'Grief/Loss': ['grief', 'loss', 'death', 'died', 'passed away', 'mourning', 'bereavement'],
    'Anger Management': ['anger', 'angry', 'rage', 'irritable', 'frustrated', 'aggression'],
    'Addiction': ['addiction', 'alcohol', 'drugs', 'substance', 'drinking', 'gambling'],
    'Eating Disorders': ['eating disorder', 'anorexia', 'bulimia', 'binge', 'food', 'weight'],
    'Sleep Issues': ['sleep', 'insomnia', 'nightmare', 'sleeping', 'tired', 'fatigue'],
    'Suicidal Thoughts': ['suicide', 'suicidal', 'kill myself', 'end my life', 'die'],
    'Social Issues': ['social', 'friends', 'lonely', 'isolation', 'loneliness', 'isolated'],
    'Work/Career': ['work', 'job', 'career', 'boss', 'colleague', 'workplace', 'employment'],
    'Parenting': ['parent', 'child', 'children', 'kids', 'son', 'daughter', 'parenting'],
    'LGBTQ+ Issues': ['gay', 'lesbian', 'transgender', 'lgbtq', 'sexuality', 'gender identity'],
    'OCD': ['ocd', 'obsessive', 'compulsive', 'ritual', 'intrusive thoughts'],
    'Bipolar': ['bipolar', 'manic', 'mania', 'mood swings'],
    'Therapy/Counseling': ['therapy', 'therapist', 'counselor', 'counseling', 'treatment']
}

# Function to categorize conversations
def categorize_conversation(text):
    """Identify categories present in the conversation"""
    if pd.isna(text):
        return []
    
    text_lower = str(text).lower()
    categories = []
    
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                categories.append(category)
                break
    
    return categories if categories else ['General/Other']

# Categorize all conversations
print("\nCategorizing conversations...")
df['categories'] = df['Context'].apply(categorize_conversation)

# Count category occurrences
all_categories = []
for cats in df['categories']:
    all_categories.extend(cats)

category_counts = Counter(all_categories)
category_df = pd.DataFrame(category_counts.items(), columns=['Category', 'Count']).sort_values('Count', ascending=False)

print("\n=== CATEGORY DISTRIBUTION ===")
print(category_df.to_string(index=False))

# Calculate percentages
category_df['Percentage'] = (category_df['Count'] / len(df) * 100).round(2)

# Create visualizations - Top 2 Charts Only
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('Mental Health Counseling Conversations - Category Analysis', fontsize=18, fontweight='bold', y=0.98)

# 1. Bar chart of top categories
ax1 = axes[0]
top_categories = category_df.head(15)
colors = sns.color_palette("husl", len(top_categories))
bars = ax1.barh(top_categories['Category'], top_categories['Count'], color=colors, edgecolor='black', linewidth=0.7)
ax1.set_xlabel('Number of Conversations', fontsize=13, fontweight='bold')
ax1.set_title('Top 15 Mental Health Categories', fontsize=14, fontweight='bold', pad=15)
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3, linestyle='--')
for i, (count, pct) in enumerate(zip(top_categories['Count'], top_categories['Percentage'])):
    ax1.text(count + 15, i, f'{count} ({pct}%)', va='center', fontsize=10, fontweight='bold')

# 2. Pie chart of top 10 categories
ax2 = axes[1]
top_10 = category_df.head(10)
colors_pie = sns.color_palette("Set3", len(top_10))
wedges, texts, autotexts = ax2.pie(top_10['Count'], labels=top_10['Category'], autopct='%1.1f%%',
                                     colors=colors_pie, startangle=90, 
                                     explode=[0.05 if i < 3 else 0 for i in range(len(top_10))],
                                     shadow=True, textprops={'fontsize': 11})
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(10)
    autotext.set_fontweight('bold')
for text in texts:
    text.set_fontsize(11)
    text.set_fontweight('bold')
ax2.set_title('Top 10 Categories Distribution', fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('mental_health_categories_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Best visualization saved as 'mental_health_categories_analysis.png'")

# Additional detailed statistics
categories_per_conv = df['categories'].apply(len)
print("\n=== DETAILED STATISTICS ===")
print(f"Average categories per conversation: {categories_per_conv.mean():.2f}")
print(f"Max categories in single conversation: {categories_per_conv.max()}")
print(f"Conversations with only 1 category: {(categories_per_conv == 1).sum()}")
print(f"Conversations with multiple categories: {(categories_per_conv > 1).sum()}")

# Create a co-occurrence matrix
print("\n=== TOP CATEGORY CO-OCCURRENCES ===")
top_cats = category_df.head(10)['Category'].tolist()
cooccurrence = np.zeros((len(top_cats), len(top_cats)))

for cats in df['categories']:
    for i, cat1 in enumerate(top_cats):
        if cat1 in cats:
            for j, cat2 in enumerate(top_cats):
                if cat2 in cats and i != j:
                    cooccurrence[i][j] += 1

# Plot co-occurrence heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(cooccurrence, annot=True, fmt='.0f', cmap='YlOrRd', 
            xticklabels=top_cats, yticklabels=top_cats,
            cbar_kws={'label': 'Co-occurrence Count'})
plt.title('Category Co-occurrence Matrix (Top 10 Categories)', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Category', fontsize=12, fontweight='bold')
plt.ylabel('Category', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('category_cooccurrence_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Co-occurrence matrix saved as 'category_cooccurrence_matrix.png'")

# Save detailed results to CSV
category_df.to_csv('mental_health_categories_summary.csv', index=False)
print("\n✓ Category summary saved as 'mental_health_categories_summary.csv'")

# Example conversations for each major category
print("\n=== SAMPLE CONVERSATIONS BY CATEGORY ===")
for category in category_df.head(5)['Category']:
    sample = df[df['categories'].apply(lambda x: category in x)].iloc[0]
    print(f"\n[{category}]")
    print(f"Context: {sample['Context'][:200]}...")
    print("-" * 80)

print("\n=== ANALYSIS COMPLETE ===")
print(f"Total categories identified: {len(category_counts)}")
print(f"Total conversations analyzed: {len(df)}")
print(f"Files created:")
print("  - mental_health_categories_analysis.png")
print("  - category_cooccurrence_matrix.png")
print("  - mental_health_categories_summary.csv")
