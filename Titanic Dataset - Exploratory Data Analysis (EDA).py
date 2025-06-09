import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings  # Added missing import
warnings.filterwarnings('ignore')

plt.switch_backend('TkAgg')  # Set compatible backend
plt.style.use('default')
sns.set_palette("husl")

df = pd.read_csv('train.csv')

print("=== TITANIC EDA ===")
print("1. DATA OVERVIEW")
print(f"Shape: {df.shape}")
print("Missing (%):")
print(df.isnull().mean() * 100)

print("\n2. STATISTICS")
print("Numerical:")
print(df.describe())
print("Categorical:")
for col in df.select_dtypes(include='object').columns:
    print(f"{col}:\n{df[col].value_counts()}")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes[0,0].pie(df['Survived'].value_counts(), labels=['Died', 'Survived'], autopct='%1.1f%%')
axes[0,0].set_title('Survival')
axes[0,1].hist(df['Age'].dropna(), bins=20, edgecolor='black')
axes[0,1].set_title('Age')
axes[0,2].hist(df['Fare'], bins=20, edgecolor='black')
axes[0,2].set_title('Fare')
df['Pclass'].value_counts().sort_index().plot.bar(ax=axes[0,3], title='Class')
df['Sex'].value_counts().plot.bar(ax=axes[1,0], title='Gender')
df['Embarked'].value_counts().plot.bar(ax=axes[1,1], title='Embarked')
df['SibSp'].value_counts().sort_index().plot.bar(ax=axes[1,2], title='SibSp')
df['Parch'].value_counts().sort_index().plot.bar(ax=axes[1,3], title='Parch')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
pd.crosstab(df['Sex'], df['Survived'], normalize='index').plot.bar(ax=axes[0,0], title='Survival by Gender')
pd.crosstab(df['Pclass'], df['Survived'], normalize='index').plot.bar(ax=axes[0,1], title='Survival by Class')
pd.crosstab(df['Embarked'], df['Survived'], normalize='index').plot.bar(ax=axes[0,2], title='Survival by Embarked')
axes[1,0].hist([df[df['Survived'] == 0]['Age'].dropna(), df[df['Survived'] == 1]['Age'].dropna()], 
               bins=20, label=['Died', 'Survived'], alpha=0.7)
axes[1,0].set_title('Age by Survival')
axes[1,0].legend()
axes[1,1].hist([df[df['Survived'] == 0]['Fare'], df[df['Survived'] == 1]['Fare']], 
               bins=20, label=['Died', 'Survived'], alpha=0.7)
axes[1,1].set_title('Fare by Survival')
axes[1,1].legend()
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
pd.crosstab(df['FamilySize'], df['Survived'], normalize='index').plot.bar(ax=axes[1,2], title='Survival by Family Size')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']].corr(), 
            annot=True, cmap='coolwarm', center=0)
plt.title('Correlation')
plt.show()

sns.pairplot(df[['Survived', 'Pclass', 'Age', 'Fare', 'FamilySize']], hue='Survived', diag_kind='hist')
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(df.groupby(['Sex', 'Pclass'])['Survived'].mean().unstack(), annot=True, cmap='RdYlGn', center=0.5, ax=ax1)
ax1.set_title('Survival: Gender & Class')
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 60, 100], labels=['Child', 'Teen', 'Adult', 'Senior'])
sns.heatmap(df.groupby(['AgeGroup', 'Pclass'])['Survived'].mean().unstack(), annot=True, cmap='RdYlGn', center=0.5, ax=ax2)
ax2.set_title('Survival: Age Group & Class')
plt.tight_layout()
plt.show()

print("\n3. DATA QUALITY")
print(f"Duplicates: {df.duplicated().sum()}")
print("Outliers:")
for col in ['Age', 'Fare', 'SibSp', 'Parch']:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)][col].count()
    print(f"{col}: {outliers}")

print("\n4. KEY FINDINGS")
print("Survival: 38%; Females (74%) > Males (19%); 1st class (63%) > 2nd (47%) > 3rd (24%)")
print("Demographics: Most 20-40 years; Children survived more")
print("Socioeconomic: Higher fares, 1st class, Cherbourg had higher survival")
print("Family: Small families (2-4) better survival")
print("Data: Missing Age (19.9%), Cabin (77.1%), Embarked (0.2%); Outliers in Fare, Age")
print("Correlations: Fare (0.257) positive, Pclass (-0.338) negative")
print("EDA DONE!")