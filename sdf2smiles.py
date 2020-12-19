import pandas as pd
from rdkit import Chem
from collections import defaultdict

sdf_sup = Chem.SDMolSupplier("E:/My Coding/Scripts/sdf files/data_lipinski.sdf")
Props = []
Props.append("Smiles")

mols = [x for x in sdf_sup if x is not None]
l = len(mols)

for i in range(l):
    for name in mols[i].GetPropNames():
        if name not in Props:
            Props.append(name)

# dictionary for storing data
param_dict = defaultdict(list)

# Read SDF , get compound parameters
sdf_sup = Chem.SDMolSupplier("E:/My Coding/Scripts/sdf files/data_lipinski.sdf")

mols = [x for x in sdf_sup if x is not None]
l = len(mols)

for i in range(l):
    # Get name
    for name in Props:
        if mols[i].HasProp(name):
            param_dict[name].append(mols[i].GetProp(name))
        else:
            param_dict[name].append(None)

df = pd.DataFrame(data=param_dict)

df['Smiles'] = [Chem.MolToSmiles(x) for x in sdf_sup if x]

df.to_csv("E:/My Coding/Scripts/sdf files/lipinski_smiles.csv", index=False)